
import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops

from utils import swap_bond_atoms

electronegativity = {
    'C': 6.27,
    'N': 7.30,
    'O': 7.54,
    'F': 10.41,
    'H': 7.18,
    'Cl': 8.30,
    'S': 6.22,
    'Br': 7.59,
    'I': 6.76,
    'P': 5.62,
    'B': 4.29,
    'Si': 4.77,
    'Se': 5.89,
    'As': 5.30,
    'Max': 10.41,
    'Min': 4.29,
    'Range': 6.12,
}

# Hardness from: https://link.springer.com/article/10.1007/s00894-013-1778-z
#https://doi.org/10.3390/i3020087
hardness = {
    'C': 5.00,
    'N': 7.23,
    'O': 6.08,
    'F': 7.01,
    'H': 6.43,
    'S': 4.14,
    'Cl': 4.68,
    'Br': 4.22,
    'I': 3.69,
    'P': 4.88,
    'B': 4.01,
    'Si': 3.38,
    'Se': 3.87,
    'As': 4.50,
    'Max': 7.23,
    'Min': 3.38,
    'Range': 3.85,
}

# from https://en.wikipedia.org/wiki/Atomic_radii_of_the_elements_(data_page)
atom_diameter = {
    'C': 75.0,
    'N': 71.0,
    'O': 63.0,
    'F': 64.0,
    'H': 32.0,
    'S': 103.0,
    'Cl': 99.0,
    'Br': 114.0,
    'I': 133.0,
    'P': 111.0,
    'B': 85.0,
    'Si': 116.0,
    'Se': 116.0,
    'As': 121.0,
    'Max': 133.0,
    'Min': 32.0,
    'Range': 101.0,
}


def one_hot(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_node_features(mol, center, args):
    ring = mol.GetRingInfo()
    mol_nodes_features = []

    for atom in mol.GetAtoms():
        node_features = []

        # Feature 1: Atomic number (#1-11)
        if args.atom_feature_element is True:
            node_features += one_hot(atom.GetSymbol(),
                                     ['B', 'C', 'N', 'O', 'F', 'Si', 'P', ('S' or 'Se'), 'Cl', 'Br', 'I'])

        # Feature 2: electronegativity (#12)
        if args.atom_feature_electronegativity is True:
            electroneg = 0.0
            if atom.GetSymbol() in electronegativity:
                electroneg = electronegativity[atom.GetSymbol()]-electronegativity['Min']
        
            electroneg /= electronegativity['Range']
            node_features += [electroneg]

        # Feature 3: hardness (#13)
        if args.atom_feature_hardness is True:
            hardn = 0.0
            if atom.GetSymbol() in hardness:
                hardn = hardness[atom.GetSymbol()] - hardness['Min']
            hardn /= hardness['Range']
            node_features += [hardn]

        # Feature 4: atom size (#14)
        if args.atom_feature_atom_size is True:
            atom_d = 0.0
            if atom.GetSymbol() in atom_diameter:
                atom_d = atom_diameter[atom.GetSymbol()] - atom_diameter['Min']
            atom_d /= atom_diameter['Range']
            node_features += [atom_d]

        # Feature 5: Hybridization (#15-17)
        if args.atom_feature_hybridization is True:
            node_features += one_hot(atom.GetHybridization(),
                                     [Chem.rdchem.HybridizationType.SP,
                                      Chem.rdchem.HybridizationType.SP2,
                                      Chem.rdchem.HybridizationType.SP3])

        # Feature 6: Aromaticity (#18)
        if args.atom_feature_aromaticity is True:
            node_features += [atom.GetIsAromatic()]

        # Feature 7: Number of rings (#19-21)
        if args.atom_feature_number_of_rings is True:
            node_features += one_hot(ring.NumAtomRings(atom.GetIdx()),
                                     [0, 1, (2 or 3 or 4 or 5 or 6)])

        # Feature 8: Ring Size (#22-25)
        if args.atom_feature_ring_size is True:
            node_features += [ring.IsAtomInRingOfSize(atom.GetIdx(), 3),
                              ring.IsAtomInRingOfSize(atom.GetIdx(), 4),
                              ring.IsAtomInRingOfSize(atom.GetIdx(), 5),
                              (ring.IsAtomInRingOfSize(atom.GetIdx(), 6) or
                               ring.IsAtomInRingOfSize(atom.GetIdx(), 7) or
                               ring.IsAtomInRingOfSize(atom.GetIdx(), 8) or
                               ring.IsAtomInRingOfSize(atom.GetIdx(), 9) or
                               ring.IsAtomInRingOfSize(atom.GetIdx(), 10))]

        # Feature 9: Number of hydrogen (#26-29)
        if args.atom_feature_number_of_Hs is True:
            node_features += one_hot(atom.GetTotalNumHs(),
                                     [0, 1, 2, (3 or 4)])

        # Feature 10: Atom formal charge (#30-32)
        if args.atom_feature_formal_charge is True:
            node_features += one_hot(atom.GetFormalCharge(),
                                     [-1, 0, 1])

        # Feature 11: IonizationCenter or not (#33)
        if int(atom.GetIdx()) == center:
            node_features += [1]
        else:
            node_features += [0]

        # Append node features to matrix
        mol_nodes_features.append(node_features)

    mol_nodes_features = np.array(mol_nodes_features)
    return torch.tensor(mol_nodes_features, dtype=torch.float)


def get_edge_features(mol, args):
    edges_features = []
    for bond in mol.GetBonds():
        edge_features1 = []
        edge_features2 = []
        # Feature 1: Bond type (#1-4)
        if args.bond_feature_bond_order is True:
            edge_features1 += one_hot(bond.GetBondTypeAsDouble(),
                                      [1, 1.5, 2, 3])
            edge_features2 += one_hot(bond.GetBondTypeAsDouble(),
                                      [1, 1.5, 2, 3])

        # Feature 2: Conjugation (#5)
        if args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is False and args.bond_feature_focused is False:
            edge_features1.append(bond.GetIsConjugated())
            edge_features2.append(bond.GetIsConjugated())

        # Feature 3: bond polarization (based on electronegativity, #6)
        if args.bond_feature_polarization is True:
            element1 = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetSymbol()
            element2 = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetSymbol()
            polarization = 0
            if element1 in electronegativity and element2 in electronegativity:
                polarization = (electronegativity[element1] - electronegativity[element2]) / electronegativity['Range']
            edge_features1 += [polarization]
            edge_features2 += [-polarization]

        # strong conjugation (with charge, #7)
        if args.bond_feature_charge_conjugation is True or args.bond_feature_focused is True:
            atom1 = bond.GetBeginAtomIdx()
            atom2 = bond.GetEndAtomIdx()
            central_atom = -1
            peripheral_atom = -1
            strongConjugation = 0
            weakConjugation = 0
            # if not conjugated, but one atom with a heteroatom and the other one in a double bond
            if bond.GetBondTypeAsDouble() == 1:
                if mol.GetAtomWithIdx(atom1).GetSymbol() == "O" or mol.GetAtomWithIdx(atom1).GetSymbol() == "N":
                    central_atom = atom2
                    peripheral_atom = atom1
                elif mol.GetAtomWithIdx(atom2).GetSymbol() == "O" or mol.GetAtomWithIdx(atom2).GetSymbol() == "N":
                    central_atom = atom1
                    peripheral_atom = atom2

                if central_atom != -1 and mol.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == -1 and mol.GetAtomWithIdx(central_atom).GetSymbol() == "C":
                    for bond2 in mol.GetBonds():
                        if bond2.GetBondTypeAsDouble() == 1:
                            continue
                        if bond2.GetBeginAtomIdx() == central_atom and (mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                        mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                            strongConjugation = 1
                            break
                        elif bond2.GetEndAtomIdx() == central_atom and (mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                        mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                            strongConjugation = 1
                            break
                if central_atom != -1 and mol.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and mol.GetAtomWithIdx(central_atom).GetSymbol() == "C":
                    for bond2 in mol.GetBonds():
                        if bond2.GetBondTypeAsDouble() == 1:
                            continue
                        if bond2.GetBeginAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 1:
                            strongConjugation = 1
                            break
                        elif bond2.GetEndAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 1:
                            strongConjugation = 1
                            break
                        if bond2.GetBeginAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0:
                            weakConjugation = 1
                            break
                        elif bond2.GetEndAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0:
                            weakConjugation = 1
                            break

            elif bond.GetBondTypeAsDouble() == 2:
                if mol.GetAtomWithIdx(atom1).GetSymbol() == "O" or mol.GetAtomWithIdx(atom1).GetSymbol() == "N":
                    central_atom = atom2
                    peripheral_atom = atom1
                elif mol.GetAtomWithIdx(atom2).GetSymbol() == "O" or mol.GetAtomWithIdx(atom2).GetSymbol() == "N":
                    central_atom = atom1
                    peripheral_atom = atom2

                if central_atom != -1 and mol.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 1 and mol.GetAtomWithIdx(central_atom).GetSymbol() == "C":
                    for bond2 in mol.GetBonds():
                        if bond2.GetBondTypeAsDouble() != 1:
                            continue
                        if bond2.GetBeginAtomIdx() == central_atom and (mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                                                        mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                            strongConjugation = 1
                            break
                        elif bond2.GetEndAtomIdx() == central_atom and (mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                                                        mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                            strongConjugation = 1
                            break
                elif central_atom != -1 and mol.GetAtomWithIdx(peripheral_atom).GetFormalCharge() == 0 and mol.GetAtomWithIdx(central_atom).GetSymbol() == "C":
                    for bond2 in mol.GetBonds():
                        if bond2.GetBondTypeAsDouble() != 1:
                            continue
                        if bond2.GetBeginAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == -1:
                            weakConjugation = 0
                            strongConjugation = 1
                            break
                        elif bond2.GetEndAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == -1:
                            weakConjugation = 0
                            strongConjugation = 1
                            break
                        if bond2.GetBeginAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetFormalCharge() == 0 \
                                and (mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "O" or
                                     mol.GetAtomWithIdx(bond2.GetEndAtomIdx()).GetSymbol() == "N"):
                            weakConjugation = 1
                        elif bond2.GetEndAtomIdx() == central_atom and mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetFormalCharge() == 0 \
                                and (mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "O" or
                                     mol.GetAtomWithIdx(bond2.GetBeginAtomIdx()).GetSymbol() == "N"):
                            weakConjugation = 1

            if args.bond_feature_conjugation is True:
                if args.bond_feature_charge_conjugation is False and strongConjugation == 1:
                    weakConjugation = 1
                #if args.bond_feature_focused is False:
                edge_features1 += [weakConjugation]
                edge_features2 += [weakConjugation]

            if args.bond_feature_charge_conjugation is True:
                edge_features1 += [strongConjugation]
                edge_features2 += [strongConjugation]

        # Append edge features to matrix (twice, per direction, 7 features)
        edges_features += [edge_features1, edge_features2]

    edges_features = np.array(edges_features)
    return torch.tensor(edges_features, dtype=torch.float)


def get_edge_info(mol_A, mol_B, center, distance_matrix):
    edge_indices = []

    # The polarization must be signed (inductive donor or acceptor). We set the first atom of the bond as the closest to the center.
    mol_A, mol_B = swap_bond_atoms(mol_A, mol_B, center, distance_matrix)

    for bond in mol_A.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices += [[i, j], [j, i]]

    edge_indices = torch.tensor(edge_indices)
    edge_indices = edge_indices.t().to(torch.long).view(2, -1)
    return edge_indices, mol_A, mol_B


def get_distance_matrix(mol):
    distance_matrix = rdmolops.GetDistanceMatrix(mol)
    return distance_matrix


def get_labels(label):
    label = np.array([label])
    return torch.tensor(label, dtype=torch.float)


def get_mol_charge(mol):
    mol_formal_charge = rdmolops.GetFormalCharge(mol)
    return torch.tensor(mol_formal_charge, dtype=torch.float)


def get_center_charge(mol, center):
    center_formal_charge = 0
    for atom in mol.GetAtoms():
        if int(atom.GetIdx()) == center:
            center_formal_charge = atom.GetFormalCharge()
    return torch.tensor(center_formal_charge, dtype=torch.float)


def from_acid_to_base(mol, center):
    base_found = False
    for atom in mol.GetAtoms():
        if int(atom.GetIdx()) == center:
            if atom.GetTotalNumHs() > 0 or atom.GetSymbol() == 'C':
                charge = atom.GetFormalCharge()
                atom.SetFormalCharge(charge - 1)
                base_found = True
                if atom.GetNumExplicitHs() > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs()-1)
            break

    # Kekulizing is causing issues with pyridinium.This step is not needed to assign aromaticity so we do all the sanitization steps but kekulization.
    # In addition, the valence is misassigned in some cases. So we also make sure to remove this step from sanitization.
    smile_base = "none"
    if base_found is True:
        Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                         Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                         Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)

        smile_base = Chem.MolToSmiles(mol)

    return base_found, mol, smile_base
