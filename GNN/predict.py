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
# To vizualize smiles, https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html

import torch
import numpy as np
import random
import os
import time
import pandas as pd
import copy
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from GNN import GNN, GNN_New
from prepare_set import generate_infersets, dump_datasets
from transfer_chirality import transfer_chirality, process_transfer_chirality_in_batches
from utils import calculate_metrics, load_data
from argParser import argsParser
from utils import find_protonation_state


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
        # initial_proposed_center = int(small_mol['Index']) + 1
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


def infer(i, small_mol, initial, ionized_smiles, ionization_states, infer_path, model_params, device, best_hypers, loss_fn, args):

    infer_dataset, ionized_smiles = generate_infersets(small_mol, i, initial, ionized_smiles, ionization_states, args)
    dump_datasets(infer_dataset, infer_path)

    # If no acid/base pair found, we exit
    if len(infer_dataset) == 0:
        infer_predicts = []
        infer_labels = []
        infer_smiles = []
        infer_smiles_base = []
        infer_mol_num = []
        infer_centers = []
        infer_proposed_centers = []
        infer_neutrals = []
        infer_ionization_states = []
        return infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, \
            infer_centers, infer_proposed_centers, infer_neutrals, infer_ionization_states, ionized_smiles

    # Loading data for training
    infer_data = load_data(args.infer_pickled)

    if args.GATv2Conv_Or_Other == "GATv2Conv":
        model_infer = GNN(feature_size=infer_dataset[0].x.shape[1],
                          edge_dim=infer_dataset[0].edge_attr.shape[1],
                          model_params=model_params)
    else:
        model_infer = GNN_New(feature_size=infer_dataset[0].x.shape[1],
                          edge_dim=infer_dataset[0].edge_attr.shape[1],
                          model_params=model_params)

    checkpoint = torch.load(args.model_dir + args.model_name, map_location=torch.device('cpu'), weights_only=True)
    model_infer.load_state_dict(checkpoint['model_state_dict'])
    model_infer.eval()

    infer_loader = DataLoader(infer_data, best_hypers["batch_size"],
                              num_workers=0, shuffle=False)

    infer_loss, infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_centers, infer_proposed_centers,\
        infer_mol_num, infer_neutral, infer_error, infer_ionization_states = \
        final_test(model=model_infer, loader=infer_loader, loss_fn=loss_fn, args=args)

    return infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
        infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles



def final_test(loader, model, loss_fn, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.mode == 'train' else "cpu")

    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    all_centers = []
    all_proposed_centers = []
    all_smiles = []
    all_smiles_base = []
    all_mol_num = []
    all_neutral = []
    all_error = []
    all_ionization_states = []
    model.p = 0
    for batch in loader:
        batch.to(device)

        with torch.no_grad():  # turns of the autograd engine. Less memory and faster

            pred = model(batch.x.float(),
                         batch.edge_index,
                         batch.edge_attr.float(),
                         batch.node_index,
                         batch.mol_formal_charge,
                         batch.center_formal_charge,
                         batch.batch)

        if pred.size()[0] > 1:
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        else:
            loss = loss_fn(pred[0], batch.y.float())
            all_preds.append(pred[0].cpu().detach().numpy())

        running_loss += loss.item()
        step += 1
        all_smiles.append(batch.smiles)
        all_smiles_base.append(batch.smiles_base)
        all_neutral.append(batch.neutral.cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
        all_centers.append(batch.ionization_center.cpu().detach().numpy())
        all_proposed_centers.append(batch.proposed_center.cpu().detach().numpy())
        all_mol_num.append(batch.mol_number.cpu().detach().numpy())
        all_error.append(batch.error.cpu().detach().numpy())
        all_ionization_states.append(batch.ionization_state)

    if len(all_preds) > 0 and all_preds[0].size > 1:
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        all_centers = np.concatenate(all_centers).ravel()
        all_proposed_centers = np.concatenate(all_proposed_centers).ravel()
        all_smiles = np.concatenate(all_smiles).ravel()
        all_smiles_base = np.concatenate(all_smiles_base).ravel()
        all_neutral = np.concatenate(all_neutral).ravel()
        all_mol_num = np.concatenate(all_mol_num).ravel()
        all_error = np.concatenate(all_error).ravel()
    elif len(all_preds) > 0:
        all_preds = np.array(all_preds[0]).ravel()
        all_labels = np.array(all_labels[0]).ravel()
        all_centers = np.array(all_centers[0]).ravel()
        all_proposed_centers = np.array(all_proposed_centers[0]).ravel()
        all_smiles = np.array(all_smiles[0]).ravel()
        all_smiles_base = np.array(all_smiles_base[0]).ravel()
        all_neutral = np.array(all_neutral[0]).ravel()
        all_mol_num = np.array(all_mol_num[0]).ravel()
        all_error = np.array(all_error[0]).ravel()

    if step == 0:
        step = 1

    if args.mode == 'train':
        calculate_metrics(all_preds, all_labels, all_mol_num, args)

    return running_loss / step, all_preds, all_labels, all_smiles, all_smiles_base, all_centers, all_proposed_centers,\
        all_mol_num, all_neutral, all_error, all_ionization_states


def export_sdf_rdkit(dataframe, output_file='molecules_prostates.sdf'):
    """
    Exports molecules from the 'Predicted pKa smiles' column of the DataFrame to a single SDF file.

    Args:
        dataframe (pd.DataFrame): DataFrame containing the 'Predicted pKa smiles' column with SMILES strings.
        output_file (str): The filename for the output SDF file."""

    smiles_column = "Predicted pKa smiles updated" if "Predicted pKa smiles updated" in dataframe.columns else "Predicted pKa smiles"
    writer = Chem.SDWriter(output_file)
    for idx, row in dataframe.iterrows():

        smiles = row.get(smiles_column)
        if not isinstance(smiles, str) or not smiles.strip():
            print(f"Skipping row {idx}: invalid or missing SMILES string.")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Could not parse SMILES on row {idx}: '{smiles}'")
            continue

        # Optionally, set a name or other properties to the molecule (will be included in the SDF)
        mol.SetProp("_Name", str(row.get('Name')))

        # Add missing (implicit) Hs as explicit Hs
        mol_with_all_h = Chem.AddHs(mol, addCoords=True)

        # Optionally generate coordinates (comment out if you don’t need them)
        AllChem.EmbedMolecule(mol_with_all_h, randomSeed=0xf00d)

        # Write to output SDF
        writer.write(mol_with_all_h)

    writer.close()
    print(f"SDF file saved to '{output_file}'.")


if __name__ == '__main__':

    csv_path = r'C:\Users\Jerome Genzling\OneDrive - McGill University\Documents\Research\pKa predictor\ACIE Submission\pKaPredictorPipeline\Datasets\trbl.csv'
    csv = pd.read_csv(csv_path)
    a,b = predict(csv)
    print('Results:')
    #add results to csv
    csv['Predicted pKa'] = a #This corresponds to the pKa value < pH that gave this protonation site
    csv['Predicted pKa smiles'] = b

    # # for smiles do transfer_chirality(orig_smiles, prot_smiles)
    # for i in range(len(csv)):
    #     orig_smiles = csv.iloc[i]['Smiles']
    #     prot_smiles = csv.iloc[i]['Predicted pKa smiles']
    #     if orig_smiles != prot_smiles:
    #         new_smiles = transfer_chirality(orig_smiles, prot_smiles)
    #         csv.loc[i, 'Predicted pKa smiles'] = new_smiles

    updated_csv = process_transfer_chirality_in_batches(csv , batch_size=50)
    print(csv)
    updated_csv.to_csv(csv_path.replace('.csv' , '_updated.csv') , index=False)
    export_sdf_rdkit(updated_csv, csv_path.replace('.csv' , '_updated.sdf'))



    # smi_path = r"C:\Users\Jerome Genzling\OneDrive - McGill University\Documents\Research\pKa predictor\ACIE Submission\pKaPredictorPipeline\Datasets\missing_compounds.smi"
    '''writer = Chem.SDWriter('test_fabio.sdf')
    smi = pd.read_csv(smi_path, header=None)
    smi.columns = ['Smiles']
    for idx, row in smi.iterrows():
        smiles = row.get('Smiles')
        if '.' in smiles:
            continue

        if not isinstance(smiles, str) or not smiles.strip():
            print(f"Skipping row {idx}: invalid or missing SMILES string.")
            continue

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"Could not parse SMILES on row {idx}: '{smiles}'")
            continue

        # Optionally, set a name or other properties to the molecule (will be included in the SDF)
        mol.SetProp("_Name", row.get('Name'))

        # Add missing (implicit) Hs as explicit Hs
        mol_with_all_h = Chem.AddHs(mol, addCoords=True)

        # Optionally generate coordinates (comment out if you don’t need them)
        AllChem.EmbedMolecule(mol_with_all_h, randomSeed=0xf00d)

        # Write to output SDF
        writer.write(mol_with_all_h)
    writer.close()'''