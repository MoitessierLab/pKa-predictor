import torch
import numpy as np
import pandas as pd
import pickle
import copy

from rdkit import Chem
from tqdm import tqdm
from torch_geometric.data import Data

from featurizer import from_acid_to_base, get_node_features, get_edge_features, get_distance_matrix, \
    get_mol_charge, get_center_charge, get_edge_info, get_labels
from utils import isDigit, swap_tensor_items, swap_tensor_values, swap_tensor_columns
from change_ionization import addHs, ionizeN, parse_smiles, find_centers


def generate_datasets(filename, train_or_test, args):
    data = pd.read_csv(filename, sep=',')
    datasets = []

    # construct a random number generator - rng
    rng = np.random.default_rng(12345)

    print('| Reading the files and computing the features...                                                                            |')

    invalid = 0
    for i, mol in tqdm(data.iterrows(), total=data.shape[0]):
        if mol['Center'] == 'C' and args.carbons_included is False:
            continue

        keep = False
        smiles_A = mol['Smiles']

        # In case of extra lines in the csv
        if smiles_A is None:
            continue

        # We check the validity of the SMILES string
        mol_test = Chem.MolFromSmiles(smiles_A, sanitize=True)

        if mol_test is None:
            print('| Invalid SMILES for mol # %-5s %91s |' % (i + 1, smiles_A))
            invalid += 1
            continue

        mol_obj_A = Chem.MolFromSmiles(smiles_A, sanitize=False)
        Chem.rdmolops.RemoveHs(mol_obj_A, sanitize=False)
        Chem.SanitizeMol(mol_obj_A, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)

        mol_obj_B = copy.deepcopy(mol_obj_A)

        if args.verbose > 0 and args.mode != 'test_with_IC':
            print('| original: %-112s |' % smiles_A)

        # Removing the hydrogen bonds (eg: [OH:15] should become O
        for j in range(len(smiles_A)):
            if j == len(smiles_A):
                break

            if smiles_A[j] == ':':
                pos = 0
                for k in range(j + 1, len(smiles_A)):
                    if isDigit(smiles_A[k]) is False:
                        pos = k
                        break
                smiles_A = smiles_A[:j] + smiles_A[pos:]
                smiles_A = smiles_A.replace('[N]', 'N').replace('[n]', 'n').replace('[O]', 'O')

        # Removing explicit hydrogens:
        smiles_A = smiles_A.replace('([H])', '')
        smiles_A = smiles_A.replace('[H]', '')
        smiles_A = smiles_A.replace('[C-]', 'C')

        negative_nitrogens = []
        smiles_A = addHs(smiles_A, mol_obj_A, mol_obj_A.GetNumAtoms(), negative_nitrogens)
        if args.verbose > 0 and args.mode != 'test_with_IC':
            print('|  revised: %-112s |' % smiles_A)

        # Get ionisation center index
        center = int(mol['Index'])

        # Prepare base from acid
        base_found, mol_obj_B, smiles_B = from_acid_to_base(mol_obj_B, center)
        if base_found is False:
            continue

        # Get node features
        node_features_A = get_node_features(mol_obj_A, center, args)
        node_features_B = get_node_features(mol_obj_B, center, args)

        # Get distance matrix (common to acids and bases)
        distance_matrix = get_distance_matrix(mol_obj_A)

        # Get adjacency info (common to acids and bases) and reorder bonds
        edge_index, mol_obj_A, mol_obj_B = get_edge_info(mol_obj_A, mol_obj_B, center, distance_matrix)

        # Get edge features
        edge_features_A = get_edge_features(mol_obj_A, args)
        edge_features_B = get_edge_features(mol_obj_B, args)
        if args.acid_or_base == "acid":
            edge_features = edge_features_A
        elif args.acid_or_base == "base":
            edge_features = edge_features_B
        elif args.acid_or_base == "both":
            if args.bond_feature_focused is False:
                edge_features = torch.cat([edge_features_A, edge_features_B], axis=1)
            # If focused, bond order of acids only and bond order 2 and 3 in single bit
            elif args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is False:
                edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], axis=1)
            elif args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is True:
                edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], axis=1)
            elif args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is True:
                edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], 1)
            elif args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is False:
                edge_features = torch.cat([edge_features_A, edge_features_B], axis=1)
        else:
            print('|----------------------------------------------------------------------------------------------------------------------------|')
            print('| ERROR: acid_or_base can only be set to acid, base or both                                                                  |')
            print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

        # Get labels info
        label = get_labels(mol['pKa'])
        error = get_labels(mol['Error'])
        mol_formal_charge = get_mol_charge(mol_obj_A)
        center_formal_charge = get_center_charge(mol_obj_A, center)

        center = move_center_in_graph(node_features_A, node_features_B, edge_index, distance_matrix, center)

        number_of_graphs = args.n_random_smiles
        if number_of_graphs == 0:
            number_of_graphs = 1

        for j in range(number_of_graphs):
            # the edge (bonds) indexes are changed (new atom numbers) but their order does not change.
            # So, no need to randomize edge_features.
            if j != 0 or args.n_random_smiles != 0:
                center = randomize_graph(node_features_A, node_features_B, edge_index, distance_matrix, center, rng)

            base_feature = 0
            base_hybrid = -1
            base_arom = -1
            base_Hs = -1
            base_charge = -1

            # Feature 1: Atomic number (#1-11)
            if args.atom_feature_element is True:
                base_feature += 11

            if args.atom_feature_electronegativity is True:
                base_feature += 1

            if args.atom_feature_hardness is True:
                base_feature += 1

            if args.atom_feature_atom_size is True:
                base_feature += 1

            # Feature 2: Hybridization (#14-16)
            if args.atom_feature_hybridization is True:
                base_hybrid = base_feature
                base_feature += 3

            # Feature 3: Aromaticity (#17)
            if args.atom_feature_aromaticity is True:
                base_arom = base_feature
                base_feature += 1

            # Feature 4: Number of rings (#18-20)
            if args.atom_feature_number_of_rings is True:
                base_feature += 3

            # Feature 5: Ring Size (#21-24)
            if args.atom_feature_ring_size is True:
                base_feature += 4

            # if args.acid_or_base == "acid" or args.acid_or_base == "both":
            if args.atom_feature_number_of_Hs is True:
                base_Hs = base_feature
                base_feature += 4

            # Feature 6: Atom formal charge (#29-31)
            if args.atom_feature_formal_charge is True:
                base_charge = base_feature
                base_feature += 3

            # The two molecules have identical features (element,...), so we concatenate
            # all of them in acids + the ones that may be different in bases (aromaticity, hybridization, number of hydrogens, charge)
            if args.acid_or_base == "acid":
                node_features = node_features_A
            elif args.acid_or_base == "base":
                node_features = node_features_B
            elif args.acid_or_base == "both":
                node_features = node_features_A

                if base_hybrid > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_hybrid:base_hybrid+3]], 1)

                if base_arom > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_arom:base_arom+1]], 1)

                if base_Hs > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_Hs:base_Hs+4]], 1)

                if base_charge > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_charge:base_charge+3]], 1)

            # Below we define the local atoms around the center (common to acids and bases)
            local_atoms = np.where(distance_matrix[center] <= args.mask_size)[0]
    
            node_index = torch.tensor(local_atoms, dtype=torch.long)
    
            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        node_index=node_index,
                        mol_formal_charge=mol_formal_charge,
                        center_formal_charge=center_formal_charge,
                        ionization_center=center+1,
                        proposed_center=center+1,
                        mol_number=i+1,
                        y=label,
                        neutral=True,
                        smiles=mol['Smiles'],
                        smiles_base=smiles_B,
                        error=error,
                        ionization_state=[])

            datasets.append(data)

    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
    print('| Invalid SMILES: %-106.0f |' % invalid)
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

    return datasets


def generate_infersets(small_mol, i, initial, ionized_smiles, ionization_states, args):

    datasets = []
    atom_idx = 0
    atom_k_idx = 0
    smiles = small_mol['Smiles']
    if 'ID' in small_mol.keys():
        name = small_mol['ID']
    elif 'Name' in small_mol.keys():
        name = small_mol['Name']
    else:
        name = 'mol'

    proposed_center = -1
    ionization_states0 = []
    invalid = 0
    if 'Index' in small_mol.keys():
        if isinstance(small_mol['Index'], list):
            proposed_center = int(small_mol['Index'][-1]) + 1
        else:
            proposed_center = int(small_mol['Index']) + 1

    original_smiles = True
    smiles_idx = 0

    # Removing explicit hydrogens:
    # Removing the hydrogen bonds (eg: [OH:15] should become O
    for j in range(len(smiles)):
        if j == len(smiles):
            break

        if smiles[j] == ':':
            pos = 0
            for k in range(j + 1, len(smiles)):
                if isDigit(smiles[k]) is False:
                    pos = k
                    break
            smiles = smiles[:j] + smiles[pos:]
            smiles = smiles.replace('[N]', 'N').replace('[n]', 'n').replace('[O]', 'O')

    # Removing explicit hydrogens:
    smiles = smiles.replace('([H])', '')
    smiles = smiles.replace('([H])', '')
    smiles = smiles.replace('[C-]', 'C')
    smiles = smiles.replace('[n]', 'n')
    # Add explicit hydrogens for terminal nitrogens
    smiles = smiles.replace('(N)', '([NH2])')

    # Checking the validity of the SMILES string
    mol_test = Chem.MolFromSmiles(smiles, sanitize=True)

    if mol_test is None and smiles != '':
        print('| Invalid SMILES: %106s |' % smiles)

    # Loading the molecule to check some properties
    mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
    Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
    Chem.SanitizeMol(mol_original, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                     Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                     Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                     catchErrors=True)

    ionizable_nitrogens, positive_nitrogens, acidic_nitrogens, negative_oxygens, acidic_oxygens, acidic_carbons,\
        nitro_nitrogens = find_centers(mol_original, i, smiles, name, initial, args)

    smiles_i = smiles
    negative_nitrogens = []
    pyridinium = []

    if initial is True:
        smiles = ionizeN(smiles, mol_original, mol_original.GetNumAtoms(), acidic_nitrogens, acidic_oxygens,
                         acidic_carbons, ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens, pyridinium, args)

    if initial is True:
        # Removing the hydrogen bonds (eg: [OH:15] should become O
        for j in range(len(smiles)):
            if j == len(smiles):
                break
            if smiles[j] == ':':
                pos = 0
                for k in range(j + 1, len(smiles)):
                    if isDigit(smiles[k]) is False:
                        pos = k
                        break
                smiles = smiles[:j] + smiles[pos:]
                smiles = smiles.replace('[N]', 'N').replace('[n]', 'n').replace('[O]', 'O')

        # neutralizing the N+'s and converting OH's, SH's and CH's into O-, S- and C-:
        j = 0
        while j < len(smiles):
            is_smiles, smiles, j, atom_idx = parse_smiles(smiles, j, atom_idx, initial, ionizable_nitrogens,
                                                          positive_nitrogens, acidic_nitrogens, negative_nitrogens,
                                                          negative_oxygens, acidic_oxygens, acidic_carbons, pyridinium, nitro_nitrogens,
                                                          False, True)

        # Adding hydrogens in the smiles string when needed and converting NH into N(-) when appropriate:
        mol_original = Chem.MolFromSmiles(smiles, sanitize=False)
        Chem.rdmolops.RemoveHs(mol_original, sanitize=False)
        Chem.SanitizeMol(mol_original, Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                         catchErrors=True)

        # Adding Hs in the smiles
        num_of_atoms = mol_original.GetNumAtoms()
        smiles = addHs(smiles, mol_original, num_of_atoms, negative_nitrogens)

        # Ionizing the activated NH's
        smiles = ionizeN(smiles, mol_original, num_of_atoms, acidic_nitrogens, acidic_oxygens, acidic_carbons,
                         ionizable_nitrogens, negative_nitrogens, negative_oxygens, nitro_nitrogens, pyridinium, args)

        if smiles != smiles_i and args.verbose > 0:
            print("|        | Revised: %-104s |" % smiles)

        if len(ionization_states0) == 0:
            ionization_states0.append(copy.deepcopy(ionizable_nitrogens))
            ionization_states0.append(copy.deepcopy(positive_nitrogens))
            ionization_states0.append(copy.deepcopy(acidic_nitrogens))
            ionization_states0.append(copy.deepcopy(negative_nitrogens))
            ionization_states0.append(copy.deepcopy(negative_oxygens))
            ionization_states0.append(copy.deepcopy(acidic_oxygens))
            ionization_states0.append(copy.deepcopy(acidic_carbons))
            ionization_states0.append(copy.deepcopy(nitro_nitrogens))

    else:
        ionization_states0 = copy.deepcopy(ionization_states)

        if args.verbose > 1:
            ionizable_nitrogens = ionization_states0[0]
            positive_nitrogens = ionization_states0[1]
            acidic_nitrogens = ionization_states0[2]
            negative_nitrogens = ionization_states0[3]
            negative_oxygens = ionization_states0[4]
            acidic_oxygens = ionization_states0[5]
            acidic_carbons = ionization_states0[6]
            nitro_nitrogens = ionization_states0[7]

    j = -1
    atom_idx = 0
    if initial is True:
        ionized_smiles = copy.deepcopy(smiles)

    smiles_A = smiles
    while j < len(smiles):

        # At the start, we identify all the acidic sites in the neutral molecule
        ionizable_nitrogens = copy.deepcopy(ionization_states0[0])
        positive_nitrogens = copy.deepcopy(ionization_states0[1])
        acidic_nitrogens = copy.deepcopy(ionization_states0[2])
        negative_nitrogens = copy.deepcopy(ionization_states0[3])
        negative_oxygens = copy.deepcopy(ionization_states0[4])
        acidic_oxygens = copy.deepcopy(ionization_states0[5])
        acidic_carbons = copy.deepcopy(ionization_states0[6])
        nitro_nitrogens = copy.deepcopy(ionization_states0[7])

        unchanged = False
        is_smiles = False

        # Now we ionize the molecule:
        if j < 0:
            j = 0

        is_smiles, smiles_A, j, atom_idx = parse_smiles(smiles, j, atom_idx, initial, ionizable_nitrogens,
                                                        positive_nitrogens, acidic_nitrogens,
                                                        negative_nitrogens, negative_oxygens, acidic_oxygens,
                                                        acidic_carbons, pyridinium, nitro_nitrogens, True, False)
        smiles_idx = j

        if is_smiles is True:

            mol_obj_A = Chem.MolFromSmiles(smiles_A, sanitize=False)
            Chem.rdmolops.RemoveHs(mol_obj_A, sanitize=False)
            Chem.SanitizeMol(mol_obj_A,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
            if mol_obj_A is None:
                continue

            mol_obj_B = copy.deepcopy(mol_obj_A)

            # Get ionisation center index
            center = atom_idx - 1

            # Prepare base from acid
            base_found, mol_obj_B, smiles_B = from_acid_to_base(mol_obj_B, center)
            if base_found is False:
                continue

            smiles_B = smiles_B.replace('[n]', 'n').replace('[N]', 'N').replace('[O]', 'O')

            # Get node features
            node_features_A = get_node_features(mol_obj_A, center, args)
            node_features_B = get_node_features(mol_obj_B, center, args)

            # Get distance matrix (common to acids and bases)
            distance_matrix = get_distance_matrix(mol_obj_A)

            # Get adjacency info (common to acids and bases) and reorder bonds
            edge_index, mol_obj_A, mol_obj_B = get_edge_info(mol_obj_A, mol_obj_B, center, distance_matrix)
            Chem.SanitizeMol(mol_obj_A,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)
            Chem.SanitizeMol(mol_obj_B,
                             Chem.SanitizeFlags.SANITIZE_FINDRADICALS | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                             Chem.SanitizeFlags.SANITIZE_SETCONJUGATION | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                             Chem.SanitizeFlags.SANITIZE_SYMMRINGS, catchErrors=True)

            # Get edge features
            if args.acid_or_base == "acid":
                edge_features_A = get_edge_features(mol_obj_A, args)
                edge_features = edge_features_A
            elif args.acid_or_base == "base":
                edge_features_B = get_edge_features(mol_obj_B, args)
                edge_features = edge_features_B
            elif args.acid_or_base == "both":
                edge_features_A = get_edge_features(mol_obj_A, args)
                edge_features_B = get_edge_features(mol_obj_B, args)
                if args.bond_feature_focused is False:
                    edge_features = torch.cat([edge_features_A, edge_features_B], axis=1)
                elif args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is False:
                    edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], axis=1)
                elif args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is True:
                    edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], axis=1)
                elif args.bond_feature_conjugation is True and args.bond_feature_charge_conjugation is True:
                    edge_features = torch.cat([edge_features_A, edge_features_B[:, 4:]], 1)
                elif args.bond_feature_conjugation is False and args.bond_feature_charge_conjugation is False:
                    edge_features = torch.cat([edge_features_A, edge_features_B], axis=1)
            else:
                print('|----------------------------------------------------------------------------------------------------------------------------|')
                print('| ERROR: acid_or_base can only be acid base or both                                                                          |')
                print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

            # Get labels info
            if args.mode == 'test':
                label = get_labels(small_mol['pKa'])
                error = get_labels(small_mol['Error'])
            else:
                label = 0
                error = 0

            # Get formal charge (always different by 1 unit between acids and bases, so no need for base charge)
            mol_formal_charge = get_mol_charge(mol_obj_A)
            center_formal_charge = get_center_charge(mol_obj_A, center)

            original_center = center
            center = move_center_in_graph(node_features_A, node_features_B, edge_index, distance_matrix, center)

            # The two molecules have identical features (element,...), so we concatenate all of them in acids + the ones
            # that may be different in bases (aromaticity, hybridization, number of hydrogens, charge)
            base_feature = 0
            base_hybrid = -1
            base_arom = -1
            base_Hs = -1
            base_charge = -1

            # Feature 1: Atomic number (#1-11)
            if args.atom_feature_element is True:
                base_feature += 11

            if args.atom_feature_electronegativity is True:
                base_feature += 1

            if args.atom_feature_hardness is True:
                base_feature += 1

            if args.atom_feature_atom_size is True:
                base_feature += 1

            # Feature 2: Hybridization (#14-16)
            if args.atom_feature_hybridization is True:
                base_hybrid = base_feature
                base_feature += 3

            # Feature 3: Aromaticity (#17)
            if args.atom_feature_aromaticity is True:
                base_arom = base_feature
                base_feature += 1

            # Feature 4: Number of rings (#18-20)
            if args.atom_feature_number_of_rings is True:
                base_feature += 3

            # Feature 5: Ring Size (#21-24)
            if args.atom_feature_ring_size is True:
                base_feature += 4

            # Feature 6: Number of hydrogen (#25-28)
            if args.atom_feature_number_of_Hs is True:
                base_Hs = base_feature
                base_feature += 4

            # Feature 7: Atom formal charge (#29-31)
            if args.atom_feature_formal_charge is True:
                base_charge = base_feature
                base_feature += 3

            if args.acid_or_base == "acid":
                node_features = node_features_A
            elif args.acid_or_base == "base":
                node_features = node_features_B
            elif args.acid_or_base == "both":
                node_features = node_features_A
                if base_hybrid > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_hybrid:base_hybrid+3]], 1)

                if base_arom > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_arom:base_arom+1]], 1)

                if base_Hs > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_Hs:base_Hs+4]], 1)

                if base_charge > -1:
                    node_features = torch.cat([node_features, node_features_B[:, base_charge:base_charge+3]], 1)

            # Below we define the local atoms around the center (common to acids and bases)
            local_atoms = np.where(distance_matrix[center] <= args.mask_size)[0]

            node_index = torch.tensor(local_atoms, dtype=torch.long)

            if args.verbose > 2:
                print("\n| Saving | smiles", smiles_A)
                print("|        | N|, N+, NH, N-, O-, OH, CH: %s %s %s %s %s %s %s"
                      % (ionizable_nitrogens, positive_nitrogens, acidic_nitrogens, negative_nitrogens, negative_oxygens,
                         acidic_oxygens, acidic_carbons))

            ionization_states = [copy.deepcopy(ionizable_nitrogens), copy.deepcopy(positive_nitrogens),
                                 copy.deepcopy(acidic_nitrogens), copy.deepcopy(negative_nitrogens),
                                 copy.deepcopy(negative_oxygens), copy.deepcopy(acidic_oxygens),
                                 copy.deepcopy(acidic_carbons), copy.deepcopy(nitro_nitrogens)]

            data = Data(x=node_features,
                        edge_index=edge_index,
                        edge_attr=edge_features,
                        node_index=node_index,
                        mol_formal_charge=mol_formal_charge,
                        center_formal_charge=center_formal_charge,
                        ionization_center=original_center+1,
                        proposed_center=proposed_center,
                        mol_number=i+1,
                        y=label,
                        neutral=True,
                        smiles=smiles_A,
                        smiles_base=smiles_B,
                        error=error,
                        ionization_state=ionization_states)

            datasets.append(data)

    return datasets, ionized_smiles


def dump_datasets(dataset, path):
    dataset_dumps = pickle.dumps(dataset)
    with open(path, "wb") as file:
        file.write(dataset_dumps)
    return


def randomize_graph(node_features_A, node_features_B, edge_index, distance_matrix, center, rng):
    tensor_size = node_features_A.size(dim=0)
    # If we have 2 atoms with one (ionization center) not movable, nothing to do
    if tensor_size == 2:
        return center
    while True:
        # We do not swap the center which should be kept at 0,0
        from_idx = int(rng.random() * (tensor_size - 1)) + 1
        to_idx = int(rng.random() * (tensor_size - 1)) + 1

        if from_idx != to_idx:
            node_features_A = swap_tensor_items(node_features_A, from_idx, to_idx)
            node_features_B = swap_tensor_items(node_features_B, from_idx, to_idx)
            edge_index = torch.clone(swap_tensor_values(edge_index, from_idx, to_idx))
            distance_matrix = swap_tensor_items(distance_matrix, from_idx, to_idx)
            distance_matrix = swap_tensor_columns(distance_matrix, from_idx, to_idx, tensor_size)
            if center == from_idx:
                center = to_idx
            elif center == to_idx:
                center = from_idx

            return center


def move_center_in_graph(node_features_A, node_features_B, edge_index, distance_matrix, center):
    if center == 0:
        return 0

    node_features_A = swap_tensor_items(node_features_A, center, 0)
    node_features_B = swap_tensor_items(node_features_B, center, 0)
    edge_index = torch.clone(swap_tensor_values(edge_index, center, 0))
    distance_matrix = swap_tensor_items(distance_matrix, center, 0)
    distance_matrix = swap_tensor_columns(distance_matrix, center, 0, node_features_A.size(dim=0))

    center = 0

    return 0
