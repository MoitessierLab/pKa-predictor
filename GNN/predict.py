# pKa and protonation state predictor
# Jerome Genzling, Ziling Luo, Ben Weiser, Nicolas Moitessier
# Department of Chemistry, McGill University
# Montreal, QC, Canada
# Some preliminary installation:
# sudo apt install python3-pip
# pip install torch
# pip install numpy
# pip install torch_geometric
# pip install pandas
# pip install rdkit
# pip install seaborn
# pip install hyperopt
# To vizualize smiles, you may use: https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html

import torch
import numpy as np
import random
import os
import time
import pandas as pd
import copy
import faulthandler

#faulthandler.enable()
from GNN import GNN
from torch_geometric.loader import DataLoader
from tqdm import tqdm
#from hyperoptimize import hyperoptimize
from plot_and_print import print_model_txt
from prepare_set import generate_datasets, dump_datasets
from argParser import argsParser
from utils import set_cuda_visible_device, load_data, find_protonation_state
from train_pKa_predictor import training, testing, inferring, testing_with_IC, infer
from usage import usage
from rdkit import Chem
from rdkit.Chem import Descriptors

def set_seed(args, seed=42):
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)


def predict(csv_file, device=None, data_path=r'../Datasets/', mode='pH', pH = 7.4, model_dir =r'..\Model',
            model_name=r'\model_4-4.pth', n_graph_layers=4, mask_size=4, infer_pickled=r'..\Datasets\pickled_data\infer_troubleshoot.pkl' ):
    #csv_file can be a csv file or a pd.dataframe with the right columns

    now = time.localtime()
    results_smiles_pred = []
    results_pka_pred = []

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    args = argsParser()
    set_seed(args)
    args.data_path = data_path
    args.infer_pickled = infer_pickled
    args.n_graph_layers = n_graph_layers
    args.mask_size = mask_size
    args.model_dir = model_dir
    args.model_name = model_name
    args.pH = pH
    args.mode = mode
    args.input = csv_file
    args.verbose = 0

    if isinstance(csv_file, str):
        #I assume it's a csv file
        infer_file = args.data_path + args.input
        data = pd.read_csv(infer_file, sep=',')
    elif isinstance(csv_file, pd.DataFrame):
        data = csv_file


    infer_path = args.infer_pickled

    device = torch.device("cpu")

    best_hypers = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler_gamma': args.scheduler_gamma,
        'model_embedding_size': args.embedding_size,
        'model_gnn_layers': args.n_graph_layers,
        'model_fc_layers': args.n_FC_layers,
        'model_dropout_rate': 0,
        'model_dense_neurons': args.model_dense_neurons,
        'model_attention_heads': args.model_attention_heads,
    }

    model_params = {k: v for k, v in best_hypers.items() if k.startswith("model_")}
    loss_fn = torch.nn.MSELoss()
    library_infer_predicts = []
    library_infer_labels = []
    library_infer_smiles = []
    library_infer_smiles_base = []
    library_infer_mol_num = []
    library_infer_centers = []
    library_infer_proposed_centers = []
    library_infer_ionization_states = []

    for i, small_mol in tqdm(data.iterrows(), total=len(data)):
        #initial_proposed_center = int(small_mol['Index']) + 1
        if Descriptors.ExactMolWt(Chem.MolFromSmiles(small_mol['Smiles']))>= 1000:
            raise ValueError(f"You're trying to predict the pKa of a species that seems too big for a small molecule (>1000 g/mol): {small_mol['Smiles']}")
        ionized_smiles = ''
        initial = True
        infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
            infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles = \
            infer(i, small_mol, initial, ionized_smiles, [], infer_path, model_params, device, best_hypers, loss_fn,
                  args)

        ionized_mol_num = i + 1
        all_infer_predicts = []
        all_infer_labels = []
        all_infer_smiles = []
        all_infer_smiles_base = []
        all_neutral = []
        all_infer_mol_num = []
        all_infer_centers = []
        all_infer_proposed_centers = []
        all_infer_ionization_states = []

        found_pKas = 2
        if len(infer_ionization_states) == 0:
            found_pKas = 0
        else:
            if len(infer_ionization_states[0][0][0]) == 0:
                found_pKas = 1

        # If there is only 1 molecule, it can only be the unchanged one
        # If we have 2 molecules, it is the unchanged + 1 site ionized. No other choice.
        # we can also have the case where we start from N- then get NH, we need another round to get NH2+
        # so we check if we still have a basic nitrogen
        if len(infer_smiles) == 1 and found_pKas < 2:
            all_infer_predicts = copy.deepcopy(infer_predicts)
            all_infer_labels = copy.deepcopy(infer_labels)
            all_infer_smiles = copy.deepcopy(infer_smiles)
            all_infer_mol_num = copy.deepcopy(infer_mol_num)
            all_infer_centers = copy.deepcopy(infer_centers)
            all_infer_proposed_centers = copy.deepcopy(infer_proposed_centers)
            all_infer_ionization_states = copy.deepcopy(infer_ionization_states)

        # in case we have more than 1 ionization center, we look for the one with the highest pKa (max_pka)
        if len(infer_smiles) > 1 or found_pKas == 2:
            max_pKa = 0
            protonation_step = 1

            # We now ionize all the sites one by one in the order of pKa's
            while max_pKa > -10:

                protonation_step += 1
                max_pKa = -11.0
                max_pKa_neutral = -11.0
                if len(infer_predicts) == 0:
                    break

                max_pKa_mol = -1
                max_pKa_mol_neutral = -1
                for j in range(len(infer_predicts)):
                    if infer_neutral[j]:
                        if infer_predicts[j] >= max_pKa_neutral:
                            max_pKa_neutral = infer_predicts[j]
                            max_pKa_mol_neutral = j

                    if infer_predicts[j] >= max_pKa:
                        max_pKa = infer_predicts[j]
                        max_pKa_mol = j

                if max_pKa_mol_neutral != -1 and max_pKa_mol == -1:
                    all_infer_predicts.append(infer_predicts[max_pKa_mol_neutral])
                    all_infer_labels.append(infer_labels[max_pKa_mol_neutral])
                    all_infer_smiles.append(infer_smiles[max_pKa_mol_neutral])
                    all_infer_smiles_base.append(infer_smiles_base[max_pKa_mol_neutral])
                    all_neutral.append(infer_neutral[max_pKa_mol_neutral])
                    all_infer_mol_num.append(infer_mol_num[max_pKa_mol_neutral])
                    all_infer_centers.append(infer_centers[max_pKa_mol_neutral])
                    all_infer_proposed_centers.append(infer_proposed_centers[max_pKa_mol_neutral])
                    all_infer_ionization_states.append(infer_ionization_states[0][max_pKa_mol_neutral])

                if max_pKa_mol != -1:
                    if 'Index' in small_mol.keys():
                        if small_mol['Smiles'] == infer_smiles[max_pKa_mol] and \
                                small_mol['Index'] == infer_centers[max_pKa_mol]:
                            break
                        small_mol['Index'] = infer_centers[max_pKa_mol]
                    else:
                        if 'Index' not in small_mol.keys():
                            small_mol['Index'] = []
                        small_mol['Index'].append(infer_centers[max_pKa_mol])

                    small_mol['Smiles'] = infer_smiles[max_pKa_mol]
                    ionization_state = infer_ionization_states[0][max_pKa_mol]
                    all_infer_predicts.append(infer_predicts[max_pKa_mol])
                    all_infer_labels.append(infer_labels[max_pKa_mol])
                    all_infer_smiles.append(infer_smiles[max_pKa_mol])
                    all_infer_smiles_base.append(infer_smiles_base[max_pKa_mol])
                    all_neutral.append(infer_neutral[max_pKa_mol])
                    all_infer_mol_num.append(infer_mol_num[max_pKa_mol])
                    all_infer_centers.append(infer_centers[max_pKa_mol])
                    all_infer_proposed_centers.append(infer_proposed_centers[max_pKa_mol])
                    all_infer_ionization_states.append(infer_ionization_states[0][max_pKa_mol])

                    # if args.verbose > 1:
                    #     print("|        | round #%2s: %-102s |" % (protonation_step, small_mol['Smiles']))
                    initial = False
                    infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
                        infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles = infer(i, small_mol, initial, ionized_smiles,
                                                                                               ionization_state,
                                                                                               infer_path, model_params,
                                                                                               device, best_hypers,
                                                                                               loss_fn, args)

                # if len(infer_predicts) == 0:
                # if args.verbose > 1:
                #     print('|        | no acid/base pair found                                                                                           |')
                # break
                # elif args.verbose > 1:
                #     print_inference(infer_predicts, infer_labels, infer_smiles, infer_mol_num, infer_centers,
                #                     initial_proposed_center, args)

            if len(infer_predicts) > 0:
                for item in range(len(infer_predicts)):
                    if len(all_infer_smiles) == 0 or all_infer_smiles[len(all_infer_smiles) - 1] != infer_smiles[item] \
                            or all_infer_centers[len(all_infer_smiles) - 1] != infer_centers[item]:
                        all_infer_predicts.append(infer_predicts[item])
                        all_infer_labels.append(infer_labels[item])
                        all_infer_smiles.append(infer_smiles[item])
                        all_infer_mol_num.append(infer_mol_num[item])
                        all_infer_centers.append(infer_centers[item])
                        all_infer_proposed_centers.append(infer_proposed_centers[item])
                        all_infer_ionization_states.append(infer_ionization_states[0][item])

        # if args.verbose > 1 and len(all_infer_smiles) > 0:
        #     print("|        | Final: %-91s----------------|" % (all_infer_smiles[0]))

        # print_inference(all_infer_predicts, all_infer_labels, all_infer_smiles, all_infer_mol_num,
        #                 all_infer_centers, initial_proposed_center, args)

        preds, labels, smiles, mol_num = find_protonation_state(all_infer_predicts, all_infer_labels, all_infer_smiles, ionized_smiles, all_infer_mol_num, ionized_mol_num, initial, args)
        if len(smiles) != 0:

            results_smiles_pred.append(smiles[0])
            results_pka_pred.append(preds[0])
        else:
            results_smiles_pred.append(small_mol['Smiles'])
            results_pka_pred.append('NaN')

        # for item in range(len(all_infer_predicts)):
        #     library_infer_predicts.append(all_infer_predicts[item])
        #     library_infer_labels.append(all_infer_labels[item])
        #     library_infer_smiles.append(all_infer_smiles[item])
        #     library_infer_mol_num.append(all_infer_mol_num[item])
        #     library_infer_centers.append(all_infer_centers[item])
        #     library_infer_proposed_centers.append(all_infer_proposed_centers[item])
        #     library_infer_ionization_states.append(all_infer_ionization_states[item])

    return results_pka_pred, results_smiles_pred


if __name__ == '__main__':
    a,b = predict('trbl.csv')
