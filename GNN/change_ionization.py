import random
import torch
import numpy as np
import os
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import Chem
from rdkit.Chem import rdmolops

from utils import whichElement

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


def addHs(smiles, mol_original, num_of_atoms, negative_nitrogens):
    j = -1
    atom_idx = 0

    while True:
        j += 1
        k = j

        if j >= len(smiles):
            break

        if atom_idx == num_of_atoms:
            break

        j, element, charge, brackets = whichElement(smiles, j)

        if element == 'none':
            continue

        hydrogen_to_remove = 0
        if atom_idx in negative_nitrogens and charge == 'none' and (element == 'NH2' or element == 'NH' or
                mol_original.GetAtomWithIdx(atom_idx).GetNumExplicitHs() + mol_original.GetAtomWithIdx(atom_idx).GetNumImplicitHs() > 0):
            charge = '-'
            hydrogen_to_remove = 1

        curr_atom = mol_original.GetAtomWithIdx(atom_idx)

        if element == 'n':
            if curr_atom.GetNumImplicitHs() == 1:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'nH' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[nH' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'nH' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[nH]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

        if element == 'N':
            if curr_atom.GetNumImplicitHs() - hydrogen_to_remove == 1:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'NH' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[NH' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'NH' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[NH]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

            if curr_atom.GetNumImplicitHs() - hydrogen_to_remove == 2:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'NH2' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[NH2' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'NH2' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[NH2]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

            if curr_atom.GetNumImplicitHs() == 3:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'NH3' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[NH3' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'NH3' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[NH3]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

        elif element == 'O':
            curr_atom = mol_original.GetAtomWithIdx(atom_idx)

            if curr_atom.GetNumImplicitHs() == 1:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'OH' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[OH' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'OH' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[OH]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

        elif element == 'S':
            curr_atom = mol_original.GetAtomWithIdx(atom_idx)

            if curr_atom.GetNumImplicitHs() == 1:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'SH' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[SH' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'SH' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[SH]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

        elif element == 'Se':
            curr_atom = mol_original.GetAtomWithIdx(atom_idx)

            if curr_atom.GetNumImplicitHs() == 1:
                if brackets is True and charge != 'none':
                    smiles = smiles[:k] + 'SeH' + charge + smiles[j + 1:]
                elif brackets is False and charge != 'none':
                    smiles = smiles[:k] + '[SeH' + charge + ']' + smiles[j + 1:]
                elif brackets is True and charge == 'none':
                    smiles = smiles[:k] + 'SeH' + smiles[j + 1:]
                elif brackets is False and charge == 'none':
                    smiles = smiles[:k] + '[SeH]' + smiles[j + 1:]
                if brackets is False:
                    j += 1
                    atom_idx += 1
                    continue

        atom_idx += 1

    return smiles


def ionizeN(smiles, mol_original, num_of_atoms, acidic_nitrogens, acidic_oxygens, acidic_carbons, ionizable_nitrogens,
            negative_nitrogens, negative_oxygens, nitro_nitrogens, pyridinium, args):
    j = -1
    atom_idx = 0

    for item in range(len(smiles)):
        j += 1
        k = j

        # We search for NH, so we stop at the second to last.
        if j >= len(smiles)-1:
            break
        if atom_idx == num_of_atoms:
            break

        j, element, charge, brackets = whichElement(smiles, j)

        if element == 'none':
            continue

        curr_atom = mol_original.GetAtomWithIdx(atom_idx)

        charged = False
        if element == 'nH' and charge == 'none':
            if curr_atom.GetNumImplicitHs() + curr_atom.GetNumExplicitHs() == 1 and curr_atom.GetFormalCharge() == 0 \
                    and atom_idx in acidic_nitrogens:

                # looking for tetrazoles
                if mol_original.GetRingInfo().IsAtomInRingOfSize(curr_atom.GetIdx(), 5):
                    nitrogen = 1
                    for bond in mol_original.GetBonds():
                        atom2 = curr_atom

                        if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                            atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                        elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                            atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                        if atom2.GetIdx() != curr_atom.GetIdx() and atom2.GetSymbol() == 'N' and \
                                atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                            nitrogen += 1

                            for bond2 in mol_original.GetBonds():
                                atom3 = curr_atom
                                if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                    atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                                elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                    atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())
                                if atom3.GetIdx() != curr_atom.GetIdx() and atom3.GetSymbol() == 'N' and \
                                        atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                    nitrogen += 1

                    if nitrogen >= 3 or curr_atom.GetIsAromatic():
                        curr_atom.SetFormalCharge(-1)
                        charged = True

            # In case of amide (NH-C=O) in 6 membered ring (aromatic if N=C-OH)
            if mol_original.GetRingInfo().IsAtomInRingOfSize(curr_atom.GetIdx(), 6):
                for bond in mol_original.GetBonds():
                    atom2 = curr_atom

                    if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                    elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                    if atom2.GetIdx() != curr_atom.GetIdx() and atom2.GetSymbol() == 'C' and atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                        for bond2 in mol_original.GetBonds():
                            atom3 = curr_atom
                            if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                            elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())
                            if atom3.GetIdx() != curr_atom.GetIdx() and (atom3.GetSymbol() == 'O' or atom3.GetSymbol() == 'S') and \
                                    atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                charged = True

            # In case of nitrogen in a pyridinium (NH-N+) in 6 membered ring
            if mol_original.GetRingInfo().IsAtomInRingOfSize(curr_atom.GetIdx(), 6):
                for bond in mol_original.GetBonds():
                    atom2 = curr_atom

                    if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                    elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                    if atom2.GetIdx() in pyridinium:
                        charged = False
                        if atom_idx in ionizable_nitrogens:
                            ionizable_nitrogens.remove(atom_idx)

            if charged:
                smiles = smiles[:k] + 'n-' + smiles[j + 1:]
                if atom_idx not in negative_nitrogens:
                    negative_nitrogens.append(atom_idx)
                if atom_idx in acidic_nitrogens:
                    acidic_nitrogens.remove(atom_idx)

        elif element == 'NH2' or element == 'NH' and charge != '-':
            acidic_nitrogens, isActivated = next_to_CO_Allyl(curr_atom, mol_original, acidic_nitrogens)

            if curr_atom.GetNumImplicitHs() + curr_atom.GetNumExplicitHs() == 1 and curr_atom.GetFormalCharge() == 0:
                # Looking for activated amides/sulfonamides
                activated_group = 0
                for bond in mol_original.GetBonds():
                    atom2 = curr_atom
                    if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                    elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                    # TODO: should we distinguish neutral vs. charged sulfones/sulfonates?
                    if atom2.GetSymbol() == 'S':
                        for bond2 in mol_original.GetBonds():

                            atom3 = curr_atom
                            if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetEndAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                            elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())

                            if atom3 != curr_atom and atom3.GetSymbol() == 'O' and \
                                    atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                activated_group += 1

                    # bound to neutral phosphonate
                    if atom2.GetSymbol() == 'P':
                        for bond2 in mol_original.GetBonds():

                            atom3 = curr_atom
                            if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetEndAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                            elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())

                            if atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2\
                                    and atom3.GetFormalCharge() == 0:
                                activated_group += 3
                            elif atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2 \
                                    and atom3.GetFormalCharge() == -1:
                                activated_group -= 2

                    if atom2.GetSymbol() == 'Br' or atom2.GetSymbol() == 'Cl' or atom2.GetSymbol() == 'F':
                        activated_group += 2

                    if atom2.GetIdx() != curr_atom.GetIdx() and atom2.GetSymbol() == 'N':
                        activated_group += 2
                        # if bound to nitro
                        nitro = 0
                        for bond2 in mol_original.GetBonds():

                            atom3 = curr_atom

                            if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetEndAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                            elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())

                            if atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                nitro += 1
                            elif atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                nitro += 1

                        if nitro == 2:
                            activated_group += 1

                    for at in range(len(pyridinium)):
                        if mol_original.GetRingInfo().AreAtomsInSameRing(atom2.GetIdx(), pyridinium[at]):
                            activated_group = 3

                    if atom2.GetIdx() != curr_atom.GetIdx() and atom2.GetSymbol() == 'O':
                        activated_group += 2

                    # bound to aromatic carbon
                    if atom2.GetSymbol() == 'C':
                        if atom2.GetIsAromatic():
                            activated_group += 1
                        else:
                            for bond2 in mol_original.GetBonds():
                                atom3 = curr_atom
                                if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                    atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                                elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                    atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())

                                if atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                    activated_group += 1
                                if atom3 != curr_atom and atom3.GetSymbol() == 'S' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                    activated_group += 2  # thioamide

                    if activated_group >= 3 or isActivated is True:
                        curr_atom.SetFormalCharge(-1)
                        charged = True

            elif curr_atom.GetNumImplicitHs() + curr_atom.GetNumExplicitHs() == 2 and curr_atom.GetFormalCharge() == 0:
                activated_group = 0
                for bond in mol_original.GetBonds():
                    atom2 = curr_atom
                    if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                    elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                    if curr_atom.GetIdx() != atom2.GetIdx():
                        for at in range(len(pyridinium)):
                            if mol_original.GetRingInfo().AreAtomsInSameRing(atom2.GetIdx(), pyridinium[at]):
                                activated_group = 3

                    if atom2.GetSymbol() == 'S':
                        for bond2 in mol_original.GetBonds():

                            atom3 = curr_atom

                            if bond2.GetBeginAtomIdx() == atom2.GetIdx() and bond2.GetEndAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetEndAtomIdx())
                            elif bond2.GetEndAtomIdx() == atom2.GetIdx() and bond2.GetBeginAtomIdx() != curr_atom.GetIdx():
                                atom3 = mol_original.GetAtomWithIdx(bond2.GetBeginAtomIdx())

                            if atom3 != curr_atom and atom3.GetSymbol() == 'O' and atom3.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                                activated_group += 2

                    if activated_group >= 3 or isActivated is True:
                        curr_atom.SetFormalCharge(-1)
                        charged = True

            if charged:
                if element == 'NH2':
                    smiles = smiles[:k] + 'NH-' + smiles[j + 1:]
                else:
                    smiles = smiles[:k] + 'N-' + smiles[j + 1:]

                if atom_idx not in negative_nitrogens:
                    negative_nitrogens.append(atom_idx)
                if atom_idx in acidic_nitrogens:
                    acidic_nitrogens.remove(atom_idx)

        # we make sure we did not charge a nitrogen aromatic with already a bond to an alkyl
        elif element == 'nH' and charge == '+':
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)

        elif element == 'n' and charge == '-':
            if atom_idx not in negative_nitrogens:
                negative_nitrogens.append(atom_idx)

        elif element == 'N' and charge == '-':
            numOfBonds = 0
            for bond in mol_original.GetBonds():
                if bond.GetBeginAtomIdx() == curr_atom.GetIdx() and mol_original.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() != 'H':
                    numOfBonds += bond.GetBondTypeAsDouble()
                elif bond.GetEndAtomIdx() == curr_atom.GetIdx() and mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() != 'H':
                    numOfBonds += bond.GetBondTypeAsDouble()
            if numOfBonds == 1:
                smiles = smiles[:k] + 'NH-' + smiles[j + 1:]
                j += 2

            if atom_idx not in negative_nitrogens:
                negative_nitrogens.append(atom_idx)

        # if n is surrounded by 3 groups (no hydrogens), the + charge should have been kept.
        elif element == 'n' and charge == 'none' and atom_idx in pyridinium:
            numOfBonds = 0
            for bond in mol_original.GetBonds():
                if bond.GetBeginAtomIdx() == curr_atom.GetIdx() and mol_original.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol() != 'H':
                    numOfBonds += bond.GetBondTypeAsDouble()
                elif bond.GetEndAtomIdx() == curr_atom.GetIdx() and mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol() != 'H':
                    numOfBonds += bond.GetBondTypeAsDouble()

            # With rdkit, in 5 membered ring such as pyrrole, all the bonds are 1.5 (even if single around nitrogens)
            # as a result, the sum of bond order is 4 (although it should be 3)
            if numOfBonds > 3.5 and mol_original.GetRingInfo().IsAtomInRingOfSize(curr_atom.GetIdx(), 6):
                if k > 0 and smiles[k] == '[':
                    smiles = smiles[:k] + 'n+' + smiles[j + 1:]
                    j += 1
                else:
                    smiles = smiles[:k] + '[n+]' + smiles[j + 1:]
                    j += 3

                if atom_idx in acidic_nitrogens:
                    acidic_nitrogens.remove(atom_idx)
                if atom_idx in ionizable_nitrogens:
                    ionizable_nitrogens.remove(atom_idx)

        # in case of nitrogen in a pyridinium (NH-N+) in 6 membered ring
        elif element == 'n' and charge == 'none':
            if mol_original.GetRingInfo().IsAtomInRingOfSize(curr_atom.GetIdx(), 6):
                for bond in mol_original.GetBonds():
                    atom2 = curr_atom

                    if bond.GetBeginAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetEndAtomIdx())
                    elif bond.GetEndAtomIdx() == curr_atom.GetIdx():
                        atom2 = mol_original.GetAtomWithIdx(bond.GetBeginAtomIdx())

                    if atom2.GetIdx() != curr_atom.GetIdx() and atom2.GetIdx() in pyridinium:
                        charged = False
                        if atom_idx in ionizable_nitrogens:
                            ionizable_nitrogens.remove(atom_idx)

        elif element == 'N' and charge == '-':
            if atom_idx not in negative_nitrogens:
                negative_nitrogens.append(atom_idx)

        elif element == 'N' and charge == 'none' and curr_atom.GetIdx() not in nitro_nitrogens:
            if curr_atom.GetDegree() == 4 and curr_atom.GetNumImplicitHs() == 0 and curr_atom.GetNumExplicitHs() == 0:
                if curr_atom.GetIdx() in ionizable_nitrogens:
                    ionizable_nitrogens.remove(curr_atom.GetIdx())
                else:
                    continue
                curr_atom.SetFormalCharge(+1)
                if k > 0 and smiles[k] == '[':
                    smiles = smiles[:k] + 'N+' + smiles[j + 1:]
                    j += 1
                else:
                    smiles = smiles[:k] + '[N+]' + smiles[j + 1:]
                    j += 3
            elif curr_atom.GetDegree() == 3 and curr_atom.GetNumImplicitHs() == 0:
                ionizable_nitrogens.append(atom_idx)

        elif element[0] == 'C' and charge == 'none' and args.carbons_included is True:
            acidic_carbons, isActivated = next_to_CO_Allyl(curr_atom, mol_original, acidic_carbons)
            charged = False
            if isActivated:
                curr_atom.SetFormalCharge(-1)
                charged = True
            if element == 'C' and smiles[k] == '[':
                if charged:
                    smiles = smiles[:k] + 'C-' + smiles[j + 1:]
            elif element == 'C':
                if charged:
                    smiles = smiles[:k] + '[C-]' + smiles[j + 1:]
                    j += 2
            elif element == 'CH2':
                if charged:
                    smiles = smiles[:k] + 'CH-' + smiles[j + 1:]
            elif element == 'C@H':
                if charged:
                    smiles = smiles[:k] + 'C-' + smiles[j + 1:]
            elif element == 'C@@H':
                if charged:
                    smiles = smiles[:k] + 'C-' + smiles[j + 1:]
            elif element == 'CH':
                if charged:
                    smiles = smiles[:k] + 'C-' + smiles[j + 1:]

        atom_idx += 1

    return smiles


def parse_smiles(smiles, j, atom_idx, initial, ionizable_nitrogens, positive_nitrogens, acidic_nitrogens,
                 negative_nitrogens, negative_oxygens, acidic_oxygens, acidic_carbons, pyridinium, nitro_nitrogens, addH, removeH):
    is_smiles = False
    smiles_A = smiles
    # In the search for acidic sites, we skip all the symbols in the SMILES string other than C, N, n and O
    # j: index of the character in the SMILES string
    # atom_idx: index of the atom
    if smiles[j] == '(' or smiles[j] == ')' or smiles[j] == ']' or smiles[j] == '[' or smiles[j] == '+' or \
            smiles[j] == '-' or smiles[j] == '/' or smiles[j] == '\\' or smiles[j] == '=' or smiles[j] == '%' or \
            smiles[j] == '@' or smiles[j] == '#' or smiles[j] == '1' or smiles[j] == '2' or smiles[j] == '3' or \
            smiles[j] == '4' or smiles[j] == '5' or smiles[j] == '6' or smiles[j] == '7' or smiles[j] == '8' or \
            smiles[j] == '9' or smiles[j] == '0' or smiles[j] == 'H' or smiles[j] == ':':
        j += 1
        return False, smiles_A, j, atom_idx

    if smiles[j] == 'c':
        atom_idx += 1
        j += 1
        return False, smiles_A, j, atom_idx

    # In case of Si and Se (S would be looked at below), Cl and Br
    if smiles[j] == 'i' or smiles[j] == 'e':
        j += 1
        return False, smiles_A, j, atom_idx

    if j < len(smiles) - 1 and smiles[j] == 'C':
        if smiles[j + 1] == 'l':
            atom_idx += 1
            j += 2
            return False, smiles_A, j, atom_idx

    if j < len(smiles) - 1 and smiles[j] == 'B':
        if smiles[j + 1] == 'r':
            atom_idx += 1
            j += 2
            return False, smiles_A, j, atom_idx
        else:
            j += 1
            return False, smiles_A, j, atom_idx

    if j < len(smiles) - 1:
        # Keep NH as is
        if (smiles[j] == 'n' or smiles[j] == 'N') and smiles[j + 1] != '+' and smiles[j + 1] != '-' and \
                smiles[j + 1] != '#' and is_smiles is False and addH is False and atom_idx in acidic_nitrogens and \
                removeH is False and atom_idx not in negative_nitrogens:
            j += 1
            atom_idx += 1
            return True, smiles_A, j, atom_idx

        # Keep carbon as is (next to C%N, S=O or C=O as identified in acidic_carbons)
        if smiles[j] == 'C' and smiles[j + 1] != '-' and smiles[j + 1] != '='\
                and is_smiles is False and (addH is False or initial is False) and atom_idx in acidic_carbons:
            if j > 0:
                if smiles[j - 1] == '=':
                    j += 1
                    atom_idx += 1
                    return False, smiles_A, j, atom_idx
            j += 1
            atom_idx += 1
            return True, smiles_A, j, atom_idx

        # Protonate a nitrogen (if no N- in the molecule, we can have O- as in carboxylates - amino acids)
        if (smiles[j] == 'n' or smiles[j] == 'N') and smiles[j + 1] != '+' and smiles[j + 1] != '-' and smiles[j + 1] != '#' and smiles[j + 1] != '@' and \
                is_smiles is False and addH is True and atom_idx not in positive_nitrogens and atom_idx in ionizable_nitrogens and \
                atom_idx not in negative_nitrogens and len(negative_nitrogens) == 0:  # TODO: the following should not be needed if the training is good
            if j > 0:
                if smiles[j - 1] == '#':
                    j += 1
                    atom_idx += 1
                    return False, smiles_A, j, atom_idx
            if smiles[j + 1] == '1' or smiles[j + 1] == '2' or smiles[j + 1] == '3' or smiles[j + 1] == '4' or smiles[j + 1] == '5' or smiles[j + 1] == '6' or \
                    smiles[j + 1] == '7' or smiles[j + 1] == '8' or smiles[j + 1] == '9':
                smiles_A = smiles[:j] + '[' + smiles[j] + 'H+' + ']' + smiles[j + 1:]
                ionizable_nitrogens.remove(atom_idx)
                if atom_idx not in positive_nitrogens:
                    positive_nitrogens.append(atom_idx)
                j += 1
                atom_idx += 1
                return True, smiles_A, j, atom_idx
            # We don't charge if there is a negative nitrogen
            elif smiles[j + 1] == 'H' and smiles[j] != 'n' and len(negative_nitrogens) == 0:
                if smiles[j + 2] == ']':
                    smiles_A = smiles[:j] + smiles[j] + 'H2+' + smiles[j + 2:]
                if smiles[j + 2] == '2':
                    smiles_A = smiles[:j] + smiles[j] + 'H3+' + smiles[j + 3:]
                ionizable_nitrogens.remove(atom_idx)
                if atom_idx not in positive_nitrogens:
                    positive_nitrogens.append(atom_idx)
                j += 1
                atom_idx += 1
                return True, smiles_A, j, atom_idx
            elif smiles[j + 1] == 'H' and smiles[j] == 'n':
                j += 1
                atom_idx += 1
                return False, smiles_A, j, atom_idx
            else:
                if smiles[j + 1] != '-' and len(negative_nitrogens) == 0:
                    smiles_A = smiles[:j] + '[' + smiles[j] + 'H+]' + smiles[j + 1:]
                    ionizable_nitrogens.remove(atom_idx)
                    if atom_idx not in positive_nitrogens:
                        positive_nitrogens.append(atom_idx)
                else:
                    smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]
                    if atom_idx in negative_nitrogens:
                        negative_nitrogens.remove(atom_idx)

            is_smiles = True

        # Protonate a negatively charged nitrogen
        # if at this stage the Hs have not yet been added (N) they should be added.
        if (smiles[j] == 'n' or smiles[j] == 'N') and smiles[j + 1] == '-' and is_smiles is False and addH is True\
                and atom_idx in negative_nitrogens:
            if smiles[j + 1] == '-':
                smiles_A = smiles[:j] + smiles[j] + 'H' + smiles[j + 2:]
            elif smiles[j + 1] == 'H':
                smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 2:]
            elif smiles[j + 1] == ')':
                smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 2:]

            is_smiles = True
            negative_nitrogens.remove(atom_idx)
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)

        if (smiles[j] == 'n' or smiles[j] == 'N') and smiles[j + 1] == 'H' and smiles[j + 2] == '2' and is_smiles is False and addH is True\
                and atom_idx in negative_nitrogens:
            if smiles[j + 1] == '-':
                smiles_A = smiles[:j] + smiles[j] + 'H' + smiles[j + 2:]
            elif smiles[j + 1] == 'H':
                smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 2:]
            elif smiles[j + 1] == ')':
                smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 2:]

            is_smiles = True
            negative_nitrogens.remove(atom_idx)
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)

        # Protonate a negatively charged nitrogen
        if smiles[j] == 'N' and smiles[j + 1] == 'H' and smiles[j + 2] == '-' and is_smiles is False and addH is True and atom_idx in negative_nitrogens:
            smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 3:]
            is_smiles = True
            negative_nitrogens.remove(atom_idx)
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)

        # Neutralize a nitrogen (N+)
        elif (smiles[j] == 'N' or smiles[j] == 'n') and smiles[j + 1] == '+' and is_smiles is False and addH is False and removeH is True and \
                atom_idx in positive_nitrogens:
            if smiles[j] == 'n' and initial is True:
                pyridinium.append(atom_idx)

            if 0 < j < len(smiles) - 2:
                if smiles[j - 1] == '[' and smiles[j + 2] == ']':
                    smiles_A = smiles[:j - 1] + smiles[j] + smiles[j + 3:]  # [N+] becomes N
                    j -= 1  # we removed the bracket before so, we have to move j to the left
                else:
                    smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]  # N+ becomes N
            else:
                smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]  # N+ becomes N
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)
            positive_nitrogens.remove(atom_idx)
            if atom_idx not in ionizable_nitrogens:
                ionizable_nitrogens.append(atom_idx)
            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Neutralize a nitrogen (NH+, NH2+, NH3+)
        elif (smiles[j] == 'N' or smiles[j] == 'n') and smiles[j + 1] == 'H' and is_smiles is False and \
                addH is False and removeH is True and atom_idx in positive_nitrogens:
            num_of_Hs = 0
            if 0 < j < len(smiles) - 3:
                if smiles[j - 1] == '[' and smiles[j + 3] == ']':
                    smiles_A = smiles[:j - 1] + smiles[j] + smiles[j + 4:]  # [NH+] becomes N
                    j -= 1  # we removed the bracket before so, we have to move j to the left
                    num_of_Hs = 0
                elif smiles[j - 1] == '[' and smiles[j + 2] == '2':
                    smiles_A = smiles[:j] + smiles[j] + 'H' + smiles[j + 4:]  # [NH2+] becomes [NH]
                    num_of_Hs = 1
                elif smiles[j - 1] == '[' and smiles[j + 2] == '3':
                    smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 4:]  # [NH3+] becomes [NH2]
                    num_of_Hs = 2
            else:
                smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]
            positive_nitrogens.remove(atom_idx)
            if atom_idx not in ionizable_nitrogens:
                ionizable_nitrogens.append(atom_idx)
            if num_of_Hs == 0 and atom_idx in acidic_nitrogens:
                acidic_nitrogens.remove(atom_idx)
            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Neutralize a nitrogen (NH+, NH2+, NH3+)
        elif (smiles[j] == 'N' or smiles[j] == 'n') and smiles[j + 1] == '@' and is_smiles is False and \
                addH is False and removeH is True and atom_idx in positive_nitrogens:
            if 0 < j < len(smiles) - 3:
                if smiles[j - 1] == '[' and smiles[j + 4] == ']':
                    smiles_A = smiles[:j - 1] + smiles[j] + smiles[j + 5:]  # [N@H+] becomes [N]
                elif smiles[j - 1] == '[' and smiles[j + 2] == '@' and smiles[j + 5] == ']':
                    smiles_A = smiles[:j - 1] + smiles[j] + smiles[j + 6:]  # [N@@H+] becomes [N]
            else:
                smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]
            positive_nitrogens.remove(atom_idx)
            if atom_idx not in ionizable_nitrogens:
                ionizable_nitrogens.append(atom_idx)
            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Ionize an oxygen
        elif (smiles[j] == 'O' or (smiles[j] == 'S' and smiles[j + 1] != 'i' and smiles[j + 1] != 'e')) and smiles[j + 1] != '-' and \
                (j == 0 or smiles[j - 1] != '=') and is_smiles is False and addH is False and removeH is True and atom_idx in acidic_oxygens:
            if 0 < j < len(smiles) - 2:
                if smiles[j - 1] == '[' and smiles[j + 2] == ']':
                    smiles_A = smiles[:j] + smiles[j] + '-' + smiles[j + 2:]
                    j += 2
                else:
                    smiles_A = smiles[:j] + '[' + smiles[j] + '-]' + smiles[j + 1:]
                    j += 2
            else:
                smiles_A = smiles[:j] + '[' + smiles[j] + '-]' + smiles[j + 1:]
                j += 1
            acidic_oxygens.remove(atom_idx)
            if atom_idx not in negative_oxygens:
                negative_oxygens.append(atom_idx)

            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Ionize a nitrogen
        elif (smiles[j] == 'n' or smiles[j] == 'N') and smiles[j + 1] != '+' and smiles[j + 1] != '-' and \
                is_smiles is False and addH is False and removeH is True and atom_idx in acidic_nitrogens:
            # we do this only if there is no charged acidic nitrogen
            if len(positive_nitrogens) == 0:
                if 0 < j < len(smiles) - 2:
                    if smiles[j - 1] == '[' and smiles[j + 2] == ']':
                        smiles_A = smiles[:j] + smiles[j] + '-' + smiles[j + 2:]
                        j += 2
                    elif smiles[j - 1] != '[':
                        smiles_A = smiles[:j] + '[' + smiles[j] + '-]' + smiles[j + 1:]
                        j += 3
                elif 0 < j < len(smiles) - 3:
                    if smiles[j - 1] == '[' and smiles[j + 2] == ']':
                        smiles_A = smiles[:j] + smiles[j] + '-' + smiles[j + 2:]
                        j += 2
                    elif smiles[j + 2] != '-':
                        smiles_A = smiles[:j] + '[' + smiles[j] + '-]' + smiles[j + 1:]
                        j += 2
                elif smiles[j + 1] != '-':
                    smiles_A = smiles[:j] + '[' + smiles[j] + '-]' + smiles[j + 1:]
                    j += 1
                if atom_idx in acidic_nitrogens:
                    acidic_nitrogens.remove(atom_idx)
                if atom_idx not in negative_nitrogens:
                    negative_nitrogens.append(atom_idx)

            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Ionize a Selenium
        elif (smiles[j] == 'S' and smiles[j + 1] == 'e') and \
                (j < len(smiles) - 2 and smiles[j + 2] != '-') and (j == 0 or smiles[j - 1] != '=') \
                and is_smiles is False and addH is False and removeH is True and atom_idx in acidic_oxygens:

            if 0 < j < len(smiles) - 3:
                if smiles[j - 1] == '[' and smiles[j + 3] == ']':
                    smiles_A = smiles[:j] + smiles[j] + 'e-' + smiles[j + 3:]
                    j += 1
                else:
                    smiles_A = smiles[:j] + '[' + smiles[j] + 'e-]' + smiles[j + 2:]
                    j += 1
            else:
                smiles_A = smiles[:j] + '[' + smiles[j] + 'e-]' + smiles[j + 2:]
                j += 1

            acidic_oxygens.remove(atom_idx)
            if atom_idx not in negative_oxygens:
                negative_oxygens.append(atom_idx)

            j += 2
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Neutralize an oxygen
        elif (smiles[j] == 'O' or (smiles[j] == 'S' and smiles[j + 1] != 'i' and smiles[j + 1] != 'e')) and \
                smiles[j + 1] == '-' and (j > 0 and smiles[j - 1] != '=') \
                and is_smiles is False and addH is True and removeH is False and atom_idx in negative_oxygens:

            if 0 < j < len(smiles) - 2:
                if smiles[j - 1] == '[' and smiles[j + 2] == ']':
                    smiles_A = smiles[:j] + smiles[j] + 'H' + smiles[j + 2:]
                else:
                    smiles_A = smiles[:j] + '[' + smiles[j] + 'H]' + smiles[j + 1:]
            else:
                smiles_A = smiles[:j] + '[' + smiles[j] + 'H]'  # + smiles[j + 2:]

            if atom_idx not in acidic_oxygens:
                acidic_oxygens.append(atom_idx)
            negative_oxygens.remove(atom_idx)

            j += 1
            atom_idx += 1
            return True, smiles_A, j, atom_idx

        # Neutralize a selenium
        elif smiles[j] == 'S' and smiles[j + 1] == 'e' and (j < len(smiles) - 2 and smiles[j + 2] == '-') and (j > 0 and smiles[j - 1] != '=') \
                and is_smiles is False and addH is True and removeH is False and atom_idx in negative_oxygens:

            if 0 < j < len(smiles) - 3:
                if smiles[j - 1] == '[' and smiles[j + 3] == ']':
                    smiles_A = smiles[:j] + smiles[j] + 'eH' + smiles[j + 3:]
                else:
                    smiles_A = smiles[:j] + '[' + smiles[j] + 'eH]' + smiles[j + 2:]

            else:
                smiles_A = smiles[:j] + '[' + smiles[j] + 'eH]'
            if atom_idx not in acidic_oxygens:
                acidic_oxygens.append(atom_idx)
            negative_oxygens.remove(atom_idx)

            j += 2
            atom_idx += 1
            return True, smiles_A, j, atom_idx

        # Protonate a negatively charged carbon
        elif smiles[j] == 'C' and smiles[j + 1] == '-' and is_smiles is False and addH is True and atom_idx in acidic_carbons:
            if smiles[j + 1] == '-':
                smiles_A = smiles[:j] + smiles[j] + smiles[j + 2:]
                smiles_A = smiles_A.replace('[C]', 'C')
            elif smiles[j + 1] == 'H':
                smiles_A = smiles[:j] + smiles[j] + 'H2' + smiles[j + 2:]

            acidic_carbons.remove(atom_idx)
            j += 2
            atom_idx += 1
            return True, smiles_A, j, atom_idx

    else:
        # Ionize an oxygen
        if (smiles[j] == 'O' or smiles[j] == 'S') and (j == 0 or smiles[j - 1] != '=') \
                and is_smiles is False and addH is False and removeH is True and atom_idx in acidic_oxygens:
            smiles_A = smiles[:j] + '[' + smiles[j] + '-]'
            acidic_oxygens.remove(atom_idx)
            if atom_idx not in negative_oxygens:
                negative_oxygens.append(atom_idx)

            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Neutralize an oxygen
        # TODO: add selenium (for now it is excluded)
        elif (smiles[j] == 'O' or smiles[j] == 'S') and smiles[j - 1] != '=' \
                and is_smiles is False and addH is True and removeH is False and atom_idx in negative_oxygens:
            smiles_A = smiles[:j] + '[' + smiles[j] + ']' + smiles[j + 2:]
            if atom_idx not in acidic_oxygens:
                acidic_oxygens.append(atom_idx)
            negative_oxygens.remove(atom_idx)

            j += 1
            atom_idx += 1
            return True, smiles_A, j, atom_idx

        # Add a nitrogen to the acidic nitrogen list
        elif (smiles[j] == 'N' or smiles[j] == 'n') and is_smiles is False and \
                addH is False and removeH is True:
            if atom_idx not in acidic_nitrogens:
                acidic_nitrogens.append(atom_idx)
            if atom_idx not in ionizable_nitrogens:
                ionizable_nitrogens.append(atom_idx)
            j += 1
            atom_idx += 1
            return False, smiles_A, j, atom_idx

        # Protonate a nitrogen
        elif (smiles[j] == 'n' or smiles[j] == 'N') and is_smiles is False:
            if j > 0:
                if smiles[j - 1] == '#':
                    j += 1
                    atom_idx += 1
                    return False, smiles_A, j, atom_idx
            smiles_A = smiles[:j] + '[' + smiles[j] + '+]'
            is_smiles = True
            if atom_idx in ionizable_nitrogens:
                ionizable_nitrogens.remove(atom_idx)

    atom_idx += 1
    j += 1
    return is_smiles, smiles_A, j, atom_idx


def find_centers(mol, i, smiles, name, initial, args):
    ionizable_nitrogens = []
    acidic_nitrogens = []
    charged_nitrogens = []
    acidic_oxygens = []
    negative_oxygens = []
    acidic_carbons = []
    nitro_nitrogens = []

    if initial is True:
        print('| %6s | original SMILES: %-80s ----------------|' % (str(i + 1), smiles + ' ' + str(name)))

    # We assume only implicit hydrogens
    for atom in mol.GetAtoms():
        isCenter = True
        number_of_bonds = 0
        number_of_unsaturations = 0
        number_of_hydrogens = 0
        # We count the number of bond on each atom. We assume implicit only. This routine may have to be revised if explicit hydrogens
        if atom.GetSymbol() == 'N' or atom.GetSymbol() == 'O' or atom.GetSymbol() == 'S' or atom.GetSymbol() == 'Se' or atom.GetSymbol() == 'As':
            number_of_hydrogens = atom.GetNumImplicitHs() + atom.GetNumExplicitHs()
            # Get the number of bonds
            for bond in mol.GetBonds():
                # Nitro
                if bond.GetBeginAtomIdx() == atom.GetIdx():
                    number_of_bonds += 1
                    if bond.GetBondTypeAsDouble() > 1.1:
                        number_of_unsaturations += 1
                elif bond.GetEndAtomIdx() == atom.GetIdx():
                    number_of_bonds += 1
                    if bond.GetBondTypeAsDouble() > 1.1:
                        number_of_unsaturations += 1
        if atom.GetSymbol() == 'N':
            # If already charged
            if atom.GetFormalCharge() == 1:
                # identifying nitro groups
                nitro = 0
                for bond in mol.GetBonds():
                    if bond.GetBeginAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                        if atom2.GetSymbol() == 'O':
                            nitro += 1
                    elif bond.GetEndAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                        if atom2.GetSymbol() == 'O':
                            nitro += 1
                if nitro == 2:
                    nitro_nitrogens.append(atom.GetIdx())
                    continue
                else:
                    # if N+ it needs to have hydrogens to be of interest
                    if atom.GetIdx() not in charged_nitrogens and number_of_hydrogens > 0:
                        charged_nitrogens.append(atom.GetIdx())
                    if number_of_hydrogens > 0 and atom.GetIdx() not in acidic_nitrogens:  # if only 1 H it will be removed when neutralizing:
                        acidic_nitrogens.append(atom.GetIdx())
            else:
                if atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and number_of_bonds <= 3:
                    if atom.GetIdx() not in ionizable_nitrogens:
                        ionizable_nitrogens.append(atom.GetIdx())
                elif atom.GetHybridization() == Chem.rdchem.HybridizationType.SP2 and number_of_hydrogens == 0:
                    nitro = 0
                    for bond in mol.GetBonds():
                        if bond.GetBeginAtomIdx() == atom.GetIdx():
                            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                            if atom2.GetSymbol() == 'O':
                                nitro += 1
                        elif bond.GetEndAtomIdx() == atom.GetIdx():
                            atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                            if atom2.GetSymbol() == 'O':
                                nitro += 1
                    if nitro == 2:
                        nitro_nitrogens.append(atom.GetIdx())
                        continue
                    # tetrazole:
                    if number_of_hydrogens == 0 and atom.GetIdx() not in ionizable_nitrogens and \
                        isTetrazole(atom, mol, acidic_nitrogens, ionizable_nitrogens, number_of_unsaturations) is False:
                            ionizable_nitrogens.append(atom.GetIdx())

                    # aniline nitrogens are considered sp2 (no unsaturations)
                    if number_of_unsaturations == 0 and atom.GetIdx() not in ionizable_nitrogens:
                        ionizable_nitrogens.append(atom.GetIdx())

                # Diazole
                if number_of_hydrogens == 1 and atom.GetIsAromatic() and atom.GetIdx() not in acidic_nitrogens:
                    for bond in mol.GetBonds():
                        # diazole
                        if bond.GetBeginAtomIdx() == atom.GetIdx():
                            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                            if atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP2 and atom.GetIsAromatic() and atom.GetSymbol() == 'N':
                                acidic_nitrogens.append(atom.GetIdx())
                                continue

                # tetrazole:
                isTetrazole(atom, mol, acidic_nitrogens, ionizable_nitrogens, number_of_unsaturations)
                # activated N-H are considered acidic
                acidic_nitrogens, isActivated = next_to_CO_Allyl(atom, mol, acidic_nitrogens)

        elif atom.GetSymbol() == 'O' or atom.GetSymbol() == 'S' or atom.GetSymbol() == 'Se' or atom.GetSymbol() == 'As':
            if number_of_bonds == 1 and number_of_hydrogens == 1 and atom.GetFormalCharge() == 0:
                isCenter = False

                for bond in mol.GetBonds():
                    # Phenol and carboxylic acids: OH next to carbonyl of phenyl (note: phenol oxygens are considered sp2 in rdkit)
                    if bond.GetBeginAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                        if atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                            isCenter = True
                            continue

                        # phosphates, sulfonates,...
                        if (atom2.GetSymbol() == 'P' or atom2.GetSymbol() == 'S') and \
                            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                            isCenter = True
                            continue
                    elif bond.GetEndAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                        if atom2.GetHybridization() == Chem.rdchem.HybridizationType.SP2:
                            isCenter = True
                            continue

                        # phosphates, sulfonates,...
                        if (atom2.GetSymbol() == 'P' or atom2.GetSymbol() == 'S') and \
                            atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3:
                            isCenter = True
                            continue

                if isCenter or atom.GetSymbol() == 'S' or atom.GetSymbol() == 'Se' or atom.GetSymbol() == 'As' and atom.GetIdx() not in acidic_oxygens:
                    acidic_oxygens.append(atom.GetIdx())

            if atom.GetFormalCharge() == -1:
                for bond in mol.GetBonds():
                    # Nitro
                    if bond.GetBeginAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                        if atom2.GetSymbol() == 'N':
                            isCenter = False
                            continue
                    elif bond.GetEndAtomIdx() == atom.GetIdx():
                        atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                        if atom2.GetSymbol() == 'N':
                            isCenter = False
                            continue
                if isCenter is True and atom.GetIdx() not in negative_oxygens:
                    negative_oxygens.append(atom.GetIdx())

        elif atom.GetSymbol() == 'C' and atom.GetHybridization() == Chem.rdchem.HybridizationType.SP3 and atom.GetTotalNumHs() > 0 \
                and args.carbons_included is True:
            # acidic if next to C=O, C%N or S=O or between 2 double bonds
            acidic_carbons, isActivated = next_to_CO_Allyl(atom, mol, acidic_carbons)

    return ionizable_nitrogens, charged_nitrogens, acidic_nitrogens, negative_oxygens, acidic_oxygens, acidic_carbons, nitro_nitrogens


def next_to_CO_Allyl(atom, mol, acidic_group):
    atom2 = atom
    found_allyl = 0
    isActivated = False

    # atom: atom considered as acidic site
    # atom2: atom directly bound
    # atom3: atom bound to atom 2
    # primary carbons should be ignored.
    if atom.GetNumImplicitHs() + atom.GetNumExplicitHs() == 3 and atom.GetSymbol() == 'C':
        return acidic_group, isActivated

    if atom.GetIsAromatic() and atom.GetSymbol() == 'C':
        return acidic_group, isActivated

    for bond in mol.GetBonds():
        found = False
        if bond.GetBeginAtomIdx() == atom.GetIdx():
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            found = True
        elif bond.GetEndAtomIdx() == atom.GetIdx():
            atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            found = True
        if found:
            found = False
            for bond2 in mol.GetBonds():
                atom3 = atom

                if bond2.GetBeginAtomIdx() == atom2.GetIdx():
                    atom3 = mol.GetAtomWithIdx(bond2.GetEndAtomIdx())
                    if atom3.GetIdx() == atom.GetIdx():
                        continue
                    phosphate = False
                    if atom3.GetIdx() != atom.GetIdx() and bond2.GetBondTypeAsDouble() > 1.1:
                        if bond2.GetBondTypeAsDouble() == 1.5:
                            found_allyl += 0.5
                        else:
                            found_allyl += 1
                        # in case of aniline on pyridine
                        if atom3.GetSymbol() == 'N' and bond2.GetBondTypeAsDouble() == 1.5:
                            found_allyl += 0.25
                            if atom3.GetFormalCharge() == 1:
                                found_allyl += 0.75

                    elif atom3.GetIdx() != atom.GetIdx() and bond2.GetBondTypeAsDouble() == 1:
                        if atom2.GetSymbol() == 'P':
                            if atom3.GetSymbol() == 'O' and (atom3.GetNumImplicitHs() + atom3.GetNumExplicitHs() == 1 or atom3.GetFormalCharge() == -1):
                                phosphate = True

                        # a negatively charged phosphate is not electron-withdrawing.
                        if phosphate:
                            found_allyl = 0

                elif bond2.GetEndAtomIdx() == atom2.GetIdx():
                    atom3 = mol.GetAtomWithIdx(bond2.GetBeginAtomIdx())
                    phosphate = False
                    if atom3.GetIdx() != atom.GetIdx() and bond2.GetBondTypeAsDouble() > 1.1:
                        # if negatively charged phosphate, it is not acidic.
                        if bond2.GetBondTypeAsDouble() == 1.5:
                            found_allyl += 0.5
                        else:
                            found_allyl += 1
                        if atom3.GetSymbol() == 'N' and bond2.GetBondTypeAsDouble() == 1.5:
                            found_allyl += 0.25
                            if atom3.GetFormalCharge() == 1:
                                found_allyl += 0.75

                    elif atom3.GetIdx() != atom.GetIdx() and bond2.GetBondTypeAsDouble() == 1:
                        # if negatively charged phosphate, it is not acidic.
                        if atom2.GetSymbol() == 'P':
                            if atom3.GetSymbol() == 'O' and (atom3.GetNumImplicitHs() + atom3.GetNumExplicitHs() == 1 or atom3.GetFormalCharge() == -1):
                                phosphate = True
                        if phosphate:
                            found_allyl = 0

        if found_allyl > 1.3 and atom.GetNumImplicitHs() + atom.GetNumExplicitHs() > 0:
            isActivated = True
            if atom.GetIdx() not in acidic_group:
                acidic_group.append(atom.GetIdx())
            break

    return acidic_group, isActivated


def isTetrazole(atom, mol, acidic_nitrogens, ionizable_nitrogens,number_of_unsaturations):
    if number_of_unsaturations != 2 or mol.GetRingInfo().IsAtomInRingOfSize(atom.GetIdx(), 5) is False or atom.GetIsAromatic() is False:
        return False

    idx1 = atom.GetIdx()
    idx2 = -1
    idx3 = -1
    idx4 = -1
    withAnH = -1
    if atom.GetTotalNumHs() == 1:
        withAnH = idx1

    for bond in mol.GetBonds():
        if bond.GetBeginAtomIdx() == idx1:
            atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
            if atom2.GetIsAromatic() and atom2.GetSymbol() == 'N' or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                if idx2 == -1:
                    idx2 = bond.GetEndAtomIdx()
                elif idx3 == -1 and bond.GetEndAtomIdx() != idx1 and bond.GetEndAtomIdx() != idx2:
                    idx3 = bond.GetEndAtomIdx()
                elif idx4 == -1 and bond.GetEndAtomIdx() != idx1 and bond.GetEndAtomIdx() != idx2 and bond.GetEndAtomIdx() != idx3:
                    idx4 = bond.GetEndAtomIdx()
                if atom2.GetTotalNumHs() == 1:
                    withAnH = atom2.GetIdx()
        elif bond.GetEndAtomIdx() == idx1:
            atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            if atom2.GetIsAromatic() and atom2.GetSymbol() == 'N' or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                if idx2 == -1:
                    idx2 = bond.GetEndAtomIdx()
                elif idx3 == -1 and bond.GetBeginAtomIdx() != idx1 and bond.GetBeginAtomIdx() != idx2:
                    idx3 = bond.GetEndAtomIdx()
                elif idx4 == -1 and bond.GetBeginAtomIdx() != idx1 and bond.GetBeginAtomIdx() != idx2 and bond.GetBeginAtomIdx() != idx3:
                    idx4 = bond.GetEndAtomIdx()
                if atom2.GetTotalNumHs() == 1:
                    withAnH = atom2.GetIdx()

    if idx2 != -1:
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() == idx2:
                atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                if atom2.GetIsAromatic() and atom2.GetSymbol() == 'N' or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                    if idx3 == -1 and bond.GetEndAtomIdx() != idx1 and bond.GetEndAtomIdx() != idx2:
                        idx3 = bond.GetEndAtomIdx()
                    elif idx4 == -1 and bond.GetEndAtomIdx() != idx1 and bond.GetEndAtomIdx() != idx2 and bond.GetEndAtomIdx() != idx3:
                        idx4 = bond.GetEndAtomIdx()
                    if atom2.GetTotalNumHs() == 1:
                        withAnH = atom2.GetIdx()
            elif bond.GetEndAtomIdx() == idx2:
                atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                if atom2.GetIsAromatic() and atom2.GetSymbol() == 'N' or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                    if idx3 == -1 and bond.GetBeginAtomIdx() != idx1 and bond.GetBeginAtomIdx() != idx2:
                        idx3 = bond.GetBeginAtomIdx()
                    elif idx4 == -1 and bond.GetBeginAtomIdx() != idx1 and bond.GetBeginAtomIdx() != idx2 and bond.GetBeginAtomIdx() != idx3:
                        idx4 = bond.GetBeginAtomIdx()
                    if atom2.GetTotalNumHs() == 1:
                        withAnH = atom2.GetIdx()

    if idx3 != -1:
        for bond in mol.GetBonds():
            if bond.GetBeginAtomIdx() == idx3:
                atom2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx())
                if (atom2.GetIsAromatic() and atom2.GetSymbol() == 'N') or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                    if idx4 == -1 and bond.GetEndAtomIdx() != idx1 and bond.GetEndAtomIdx() != idx2 and bond.GetEndAtomIdx() != idx3:
                        idx4 = bond.GetEndAtomIdx()
                if atom2.GetTotalNumHs() == 1:
                    withAnH = atom2.GetIdx()
            elif bond.GetEndAtomIdx() == idx3:
                atom2 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
                if (atom2.GetIsAromatic() and atom2.GetSymbol() == 'N') or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetTotalNumHs() == 1) or \
                    (mol.GetRingInfo().IsAtomInRingOfSize(atom2.GetIdx(), 5) and atom2.GetFormalCharge() == -1):
                    if idx4 == -1 and bond.GetBeginAtomIdx() != idx1 and bond.GetBeginAtomIdx() != idx2 and bond.GetBeginAtomIdx() != idx3:
                        idx4 = bond.GetBeginAtomIdx()
                if atom2.GetTotalNumHs() == 1:
                    withAnH = atom2.GetIdx()

    if idx4 != -1:
        if idx1 not in acidic_nitrogens and idx1 == withAnH:
            acidic_nitrogens.append(idx1)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)
        if idx2 not in acidic_nitrogens and idx2 == withAnH:
            acidic_nitrogens.append(idx2)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)

        if idx3 not in acidic_nitrogens and idx3 == withAnH:
            acidic_nitrogens.append(idx3)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)
        if idx4 not in acidic_nitrogens and idx4 == withAnH:
            acidic_nitrogens.append(idx4)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)

        if idx1 not in acidic_nitrogens and mol.GetAtomWithIdx(idx1).GetFormalCharge() == -1:
            ionizable_nitrogens.append(idx1)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)
        if idx2 not in acidic_nitrogens and mol.GetAtomWithIdx(idx2).GetFormalCharge() == -1:
            ionizable_nitrogens.append(idx2)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)
        if idx3 not in acidic_nitrogens and mol.GetAtomWithIdx(idx3).GetFormalCharge() == -1:
            ionizable_nitrogens.append(idx3)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
            if idx4 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx4)

        if idx4 not in acidic_nitrogens and mol.GetAtomWithIdx(idx4).GetFormalCharge() == -1:
            ionizable_nitrogens.append(idx4)
            if idx2 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx2)
            if idx3 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx3)
            if idx1 in ionizable_nitrogens:
                ionizable_nitrogens.remove(idx1)
        return True
    return False
