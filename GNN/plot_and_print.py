

import random
import torch
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem

from GNN import GNN, GNN_New
from utils import find_protonation_state, compute_mae, search

seed = 42
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False
np.random.seed(seed)
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def plot_figure1(train_loss_all, test_loss_all, args):
    plt.plot(train_loss_all, label="training")
    plt.plot(test_loss_all, label="validation")
    plt.legend()
    plt.ylim(0.2, 2)
    plt.grid(axis='y')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(args.output + 'epochs_focused.pdf')


def plot_figure2(train_labels, train_predicts, test_labels, test_predicts, args):
    sns.set(color_codes=True)
    sns.set_style("white")

    plt.clf()
    ax = sns.regplot(x=train_labels, y=train_predicts, scatter_kws={'alpha': 0.4})
    ax.annotate("$R^2$= {:.3f}".format(r2_score(train_labels, train_predicts)), (0, 1))
    ax.set_xlabel('Experimental pKa', fontsize='large', fontweight='bold')
    ax.set_ylabel('Predicted pKa', fontsize='large', fontweight='bold')
    ax.figure.set_size_inches(6, 6)
    ax.set(xlim=(-5, 20), ylim=(-5, 20))
    plt.grid(color='green', linestyle='--', linewidth=0.5)
    plt.savefig(args.output + 'train-obs_vs_pred.pdf')

    plt.clf()
    ax = sns.regplot(x=test_labels, y=test_predicts, scatter_kws={'alpha': 0.4})
    ax.annotate("$R^2$= {:.3f}".format(r2_score(test_labels, test_predicts)), (0, 1))
    ax.set_xlabel('Experimental pKa', fontsize='large', fontweight='bold')
    ax.set_ylabel('Predicted pKa', fontsize='large', fontweight='bold')
    ax.figure.set_size_inches(6, 6)
    plt.savefig(args.output + 'test-obs_vs_pred.pdf')


def plot_figure3(train_labels, train_predicts, test_labels, test_predicts, args):
    sns.set(color_codes=True)
    sns.set_style("white")
    plt.figure(figsize=(6, 6))
    ax1 = plt.subplot()
    sns.histplot(compute_mae(train_predicts, train_labels), ax=ax1, color='maroon', stat='density',
                 kde=True, fill=False)

    ax1.set_title("MAE")
    plt.tight_layout()
    plt.figure(figsize=(6, 6))

    ax1 = plt.subplot()
    sns.histplot(compute_mae(test_predicts, test_labels), ax=ax1, color='green', stat='density',
                 kde=True, fill=False)

    plt.legend(labels=['train', 'test'])
    plt.savefig(args.output + '_MAE.pdf')


def print_results(train_predicts, train_labels, train_smiles, train_mol_num, train_error, test_predicts,
                  test_labels, test_smiles, test_mol_num, test_error, args):
    train_difference = []
    for i in range(len(train_predicts)):
        train_difference.append(train_predicts[i]-train_labels[i])
    index = [i for i, v in enumerate(train_difference) if abs(v) > 1.5]
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Training set: %5s out of %7s with error > 1.5                                                                         |'
          % (len(index), len(train_smiles)))
    print('|--------------------------------------------------                                                                          |')
    for i in index:
        print('| %5.0f | %5.2f | %5.2f | %4.2f | %-91s |' % (train_mol_num[i], train_labels[i],
                                                                    train_predicts[i], train_error[i],
                                                                    train_smiles[i]))
    
    with open(args.output + 'train_results.csv', 'w') as f:
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')
        f.write('| mol # | obs.  | pred. | SMILES                                                                                             |\n')
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')
        for i in range(len(train_labels)):
            f.write('| %5.0f | %5.2f | %5.2f | %4.2f | %-91s |\n' % (train_mol_num[i], train_labels[i],
                                                                            train_predicts[i], train_error[i], train_smiles[i]))
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')

    test_difference = []
    for i in range(len(test_predicts)):
        test_difference.append(test_predicts[i]-test_labels[i])
    index = [i for i, v in enumerate(test_difference) if abs(v) > 1.5]
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Testing set: %4s out of %4s with error > 1.5                                                                             |'
          % (len(index), len(test_smiles)))
    print('|--------------------------------------------------                                                                          |')

    for i in index:
        print('| %5.0f | %5.2f | %5.2f | %4.2f | %-71s |' % (test_mol_num[i], test_labels[i],
                                                             test_predicts[i], test_error[i], test_smiles[i]))
    
    with open(args.output + 'test_results.csv', 'w') as f:
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')
        f.write('| mol # | obs.  | pred. | SMILES                                                                                             |\n')
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')
        for i in range(len(test_predicts)):
            f.write('| %5.0f | %5.2f | %5.2f | %4.2f | %-71s |\n' % (test_mol_num[i], test_labels[i],
                                                                     test_predicts[i], test_error[i], test_smiles[i]))
        f.write('|----------------------------------------------------------------------------------------------------------------------------|\n')

    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)


def print_results_test(predicts, labels, smiles, centers, proposed_centers, mol_nums, args):

    seen = []
    pred = []
    label = []
    mol_num = []
    smile = []
    proposed_center = []
    center = []
    for i in range(len(predicts)):
        # print("Molecule ", mol_num[i], y_pred[i], y_true[i])
        if search(mol_nums[i], seen) is False:
            seen.append(mol_nums[i])
            pred.append(predicts[i])
            label.append(labels[i])
            smile.append(smiles[i])
            proposed_center.append(proposed_centers[i])
            mol_num.append(mol_nums[i])
            center.append(centers[i])
        else:
            idx = seen.index(mol_nums[i])
            if abs(pred[idx]-label[idx]) > abs(predicts[i]-labels[i]):
                pred[idx] = predicts[i]
                center[idx] = centers[i]

    test_difference = []
    for item in range(len(pred)):
        test_difference.append(pred[item] - label[item])

    index = [i for i, v in enumerate(test_difference) if abs(v) > 1.0]
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Training set: %4s out of %4s with error > 1.0                                                                            |'
          % (len(index), len(test_difference)))
    print("|----------------------------------------------------------------------------------------------------------------------------|")
    print('| mol #  | SMILES                                                                            | center        |      pKa      |')
    print('|        |                                                                                   |  obs. | pred. |  obs. | pred. |')
    for i in index:
        print('| %6.0f | %-81s |  %3s  |  %3s  | %5.2f | %5.2f |' % (mol_num[i], smile[i], proposed_center[i],
                                                                     center[i], label[i], pred[i]))
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)


def print_inference(preds, labels, smiles, ionized_smiles, mol_num, ionized_mol_num, centers, proposed_center, initial, args):
    if args.mode == 'pH':
        preds, labels, smiles, mol_num = find_protonation_state(preds, labels, smiles, ionized_smiles, mol_num, ionized_mol_num, initial, args)

    if args.mode == 'test':
        for i in range(len(preds)):
            print('| %6.0f | %-81s |  %3s  |  %3s  | %5.2f | %5.2f |' % (mol_num[i], smiles[i], proposed_center,
                                                                         centers[i], labels[i], preds[i]))
    else:
        if len(centers) == 0:
            print('| %6.0f | %-88s | %6s | %13s |' % (ionized_mol_num, ionized_smiles, 'n/a', '> 14'))
        else:
            for i in range(len(preds)):
                if preds[i] == 14:
                    if centers == []:
                        print('| %6.0f | %-88s | %6s | %13s |' % (0, smiles[i], 'None', '>14'))
                    else:
                        print('| %6.0f | %-88s | %6s | %13s |' % (mol_num[i], smiles[i], centers[i], '>14'))
                else:
                    print('| %6.0f | %-88s | %6s | %13.2f |' % (mol_num[i], smiles[i], centers[i], preds[i]))


    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)


def print_model_txt(hypers, args):
    if args.load_model != 'none':
        print('| Writing the model                                                                                                          |')
        print('|----------------------------------------------------------------------------------------------------------------------------|')
        model_params = {k: v for k, v in hypers.items() if k.startswith("model_")}

        if args.GATv2Conv_Or_Other == "GATv2Conv":
            trained_model = GNN(feature_size=hypers['model_node_feature_size'],
                                edge_dim=hypers['model_edge_feature_size'],
                                model_params=model_params)
        else:
            trained_model = GNN_New(feature_size=hypers['model_node_feature_size'],
                                edge_dim=hypers['model_edge_feature_size'],
                                model_params=model_params)

        checkpoint = torch.load(args.model_dir + args.load_model, map_location=torch.device('cpu'))
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        with open(args.output + args.model_txt_file, 'w') as f:
            for name, param in trained_model.named_parameters():
                f.write(name)
                f.write('\n')
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                else:
                    np.savetxt(f, param.data, fmt='%12.8f')

            # below if for running_mean and running_var
            for name, param in trained_model.named_buffers():
                f.write(name)
                f.write('\n')
                tensor_shape = list(map(int, param.data.shape))
                tensor_list = list(tensor_shape)
                tensor_dim = len(list(tensor_shape))
                f.write(str(tensor_dim))
                f.write('\n')
                for dim in range(tensor_dim):
                    f.write(str(tensor_list[dim]))
                    f.write('\n')

                # savetxt cannot take tensors with more than 2 dimensions
                if len(list(tensor_shape)) > 2:
                    for slice_2d in param.data:
                        np.savetxt(f, slice_2d, fmt='%12.8f')
                elif len(list(tensor_shape)) > 0:
                    np.savetxt(f, param.data, fmt='%12.8f')
