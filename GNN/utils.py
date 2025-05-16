import copy
import random
import torch
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem


def set_cuda_visible_device(ngpus):
    import subprocess
    import os
    empty = []
    for i in range(8):
        command = 'nvidia-smi -i '+str(i)+' | grep "0%" | wc -l'
        output = subprocess.check_output(command, shell=True).decode("utf-8")
        if int(output) == 1:
            empty.append(i)
    if len(empty) < ngpus:
        print('| available gpus are less than required')
        exit(-1)
    cmd = ''
    for i in range(ngpus):        
        cmd += str(empty[i])+','
    return cmd


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_data(file_name):
    with open(file_name, "rb") as f:
        conts = f.read()

    return pickle.loads(conts)


def compute_mae(data, ref):
    mae = []
    for i in range(len(data)):
        mae.append((ref[i]-data[i]))
    return mae


def search(item, list_items):
    for i in range(len(list_items)):
        if list_items[i] == item:
            return True
    return False


def calculate_metrics(y_pred, y_true, mol_num, args):
    seen = []
    pred = []
    label = []

    for i in range(len(y_pred)):
        if args.mode == 'train':
            pred.append(y_pred[i])
            label.append(y_true[i])
        else:
            if search(mol_num[i], seen) is False:
                seen.append(mol_num[i])
                pred.append(y_pred[i])
                label.append(y_true[i])
            else:
                idx = seen.index(mol_num[i])
                if abs(pred[idx]-label[idx]) > abs(y_pred[i]-y_true[i]):
                    pred[idx] = y_pred[i]

    mae = mean_absolute_error(pred, label)
    mse = mean_squared_error(pred, label)

    rmse = mean_squared_error(pred, label, squared=False)
    r2 = r2_score(pred, label)
    print("| Number of molecules:  %-101.0f|" % len(pred))
    print("| MAE:                  %-101.3f|" % mae)
    print("| MSE:                  %-101.3f|" % mse)
    print("| RMSE:                 %-101.3f|" % rmse)
    print("| R2:                   %-101.3f|" % r2)


def find_protonation_state(predicts, labels, smiles, ionized_smiles, mol_num, ionized_mol_num, initial, args):
    pH_predicts = []
    pH_labels = []
    pH_smiles = []
    pH_mol_num = []
    temp_predicts = []
    temp_labels = []
    temp_smiles = []
    temp_mol_num = []

    unique_mol_num = list(dict.fromkeys(mol_num))
    if len(unique_mol_num) == 0:
        pH_predicts.append(14.0)
        pH_labels.append(0)
        pH_smiles.append(ionized_smiles)
        pH_mol_num.append(mol_num)
        return pH_predicts, pH_labels, pH_smiles, pH_mol_num
    for count in range(len(unique_mol_num)):
        temp_predicts.clear()
        temp_labels.clear()
        temp_smiles.clear()
        temp_mol_num.clear()
        temp_pH = 100
        found = -1

        for i in range(len(predicts)):
            if args.mode == "pH" or mol_num[i] == count + 1:
                temp_predicts.append(predicts[i])
                temp_labels.append(predicts[i])
                temp_smiles.append(smiles[i])
                temp_mol_num.append(mol_num[i])

        # If there is no state with a pKa higher than the requested pH, we use the fully ionized molecule
        for i in range(len(temp_predicts)):
            # the first pKa value is the one selected to start
            #if temp_pH > 99.5:
            #    found = 99
            #    temp_pH = 99
            # if the pKa is low but still higher (closer to user defined pH) that the one previously saved
            # if temp_predicts[i] > temp_pH and temp_pH < args.pH:
            # No longer need this condition because the minimum pka may not be the last protonation state, i.e not the one we want
            # ex: O=C1N[C@@H](c2cccs2)C(=O)N1C[C@H]1CCSC1
            #     found = i
            #     temp_pH = temp_predicts[i]
            # if pKa values are higher than the user defined pH, we take the lowest
            if temp_predicts[i] >= args.pH:
                found = i
                # temp_pH = temp_predicts[i]

        if found == -1 and len(predicts) == 0:
            pH_predicts.append(14.0)
            pH_labels.append(temp_labels[0])
            pH_smiles.append(ionized_smiles)
            pH_mol_num.append(temp_mol_num[0])
        elif found == -1:
            pH_predicts.append(predicts[0])
            pH_labels.append(temp_labels[0])
            pH_smiles.append(ionized_smiles)
            pH_mol_num.append(temp_mol_num[0])
        else:
            pH_predicts.append(temp_predicts[found])
            pH_labels.append(temp_labels[found])
            pH_smiles.append(temp_smiles[found])
            pH_mol_num.append(temp_mol_num[found])

    return pH_predicts, pH_labels, pH_smiles, pH_mol_num


def isDigit(char):
    if char == '0' or char == '1' or char == '2' or char == '3' or char == '4' or char == '5' or char == '6' or char == '7' or char == '8' or char == '9':
        return True

    return False


def whichElement(smiles, j):
    if smiles[j] == 'F' or smiles[j] == 'I' or smiles[j] == 'P':
        return j, smiles[j], 'none', 'none'

    char = 'none'
    char2 = 'none'
    char3 = 'none'
    char4 = 'none'
    charge = 'none'
    brackets = False
    if j < len(smiles) - 1:
        char = smiles[j + 1]
    if j < len(smiles) - 2:
        char2 = smiles[j + 2]
    if j < len(smiles) - 3:
        char3 = smiles[j + 3]
    if j < len(smiles) - 4:
        char4 = smiles[j + 4]
    if j > 0 and smiles[j - 1] == '[':
        brackets = True

    if smiles[j] == 'N' or smiles[j] == 'O' or smiles[j] == 'n' or smiles[j] == 'o' or \
            (smiles[j] == 'C' and char != 'l'):
        element = smiles[j]
        if char == 'H':
            if char2 == '2' or char2 == '3':
                j += 2
                if char3 == '+' or char3 == '-':
                    charge = char3
                    j += 1
                return j, element + char + char2, charge, brackets
            else:
                j += 1
                if char2 == '+' or char2 == '-':
                    charge = char2
                    j += 1
                return j, element + char, charge, brackets
        elif char == '@':
            if char2 == 'H':
                j += 2
                if char3 == '+' or char3 == '-':
                    charge = char3
                    j += 1
                return j, element + char + char2, charge, brackets
            if char2 == ']':
                j += 2
                return j, element + char, charge, brackets
            elif char2 == '@':
                if char3 == 'H':
                    j += 3
                    # print("utils 460", element, char, char2, char3)
                    if char4 == '+' or char4 == '-':
                        charge = char4
                        j += 1
                    return j, element + char + char2 + char3, charge, brackets
                if char3 == ']':
                    j += 3
                    return j, element + char + char2, charge, brackets
            else:
                if char2 == '+' or char2 == '-':
                    charge = char2
                    j += 1
                return j, element + char, charge, brackets
        elif char == '+' or char == '-':
            charge = char
            j += 1
            return j, element, charge, brackets
        else:
            return j, smiles[j], charge, brackets

    if smiles[j] == 'c':
        return j, 'C', charge, brackets

    if smiles[j] == 'C' and char == 'l':
        if char2 == '-':
            charge = '-'
            j += 1
        j += 1
        return j, 'Cl', charge, brackets

    if smiles[j] == 'S' or smiles[j] == 's':
        if char == 'e':
            j += 1
            return j, 'Se', charge, brackets
        elif char == 'i':
            j += 1
            return j, 'Si', charge, brackets
        else:
            if char == '-':
                charge = '-'
                j += 1
            return j, 'S', charge, brackets

    if smiles[j] == 'B':
        if char == 'r':
            if char2 == '-':
                charge = '-'
                j += 1
            j += 1
            return j, 'Br', charge, brackets
        else:
            return j, 'B', charge, brackets

    if smiles[j] == 'A':
        if char == 's':
            j += 1
            return j, 'As', charge, brackets
        else:
            return j, 'Al', charge, brackets
    return j, 'none', charge, brackets


def swap_tensor_items(original_tensor, from_item_idx, to_item_idx):
    new_tensor = original_tensor
    new_tensor[[from_item_idx, to_item_idx], ] = original_tensor[[to_item_idx, from_item_idx], ]
    return new_tensor


def swap_tensor_columns(original_tensor, from_idx, to_idx, tensor_size):
    new_tensor = original_tensor
    new_tensor[:, [from_idx, to_idx]] = original_tensor[:, [to_idx, from_idx]]
    return new_tensor


def swap_tensor_values(original_tensor, from_idx, to_idx):
    new_tensor = original_tensor
    # we replace old indexes with new indexes.
    # first we look for old indexes and search where they are
    indexes_old = (original_tensor == from_idx)
    new_tensor[indexes_old, ] = 999

    indexes_old = (original_tensor == to_idx)
    new_tensor[indexes_old, ] = from_idx

    indexes_old = (original_tensor == 999)
    new_tensor[indexes_old, ] = to_idx

    return new_tensor


def average(predicts, labels, smiles, mol_num, error, args):
    predicts_average = []
    labels_average = []
    smiles_average = []
    mol_num_average = []
    error_average = []
    count_average = []

    for i in range(len(mol_num)):
        if mol_num[i] in mol_num_average:
            idx = mol_num_average.index(mol_num[i])
            count_average[idx] = float(count_average[idx]) + 1.0
            predicts_average[idx] = float(predicts_average[idx]) + float(predicts[i])
        else:
            mol_num_average.append(mol_num[i])
            count_average.append(1.0)
            labels_average.append(labels[i])
            smiles_average.append(smiles[i])
            error_average.append(error[i])
            predicts_average.append(predicts[i])

    for i in range(len(mol_num_average)):
        predicts_average[i] = float(predicts_average[i])/float(count_average[i])

    return predicts_average, labels_average, smiles_average, mol_num_average, error_average


def swap_bond_atoms(mol_A, mol_B, center, distance_matrix):
    mw_A = Chem.RWMol(mol_A)  # The bonds are reordered when doing this...
    mw_B = Chem.RWMol(mol_B)  # The bonds are reordered when doing this...

    # now we have a rewritable molecule, we switch atoms in their bonds: the closer to the center is the first atom.
    # First we had the corresponding bonds and remove the old ones.
    toBeRemoved = []
    for i in range(mol_A.GetNumBonds()):
        bond_A = mol_A.GetBondWithIdx(i)
        bond_B = mol_B.GetBondWithIdx(i)
        atom1 = bond_A.GetBeginAtomIdx()
        atom2 = bond_A.GetEndAtomIdx()

        if distance_matrix[atom1][center] > distance_matrix[atom2][center]:
            mw_A.RemoveBond(atom1, atom2)
            mw_B.RemoveBond(atom1, atom2)
            mw_A.AddBond(atom2, atom1, bond_A.GetBondType())
            mw_B.AddBond(atom2, atom1, bond_B.GetBondType())

            i -= 1

    return mw_A, mw_B


# Moving optimizer to device. ( https://github.com/pytorch/pytorch/issues/8741)
def optimizer_to(optim, device):
    for param in optim.state.values():
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)
