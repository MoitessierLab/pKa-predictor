import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFMCS
import time
from tqdm import tqdm  # For progress tracking


def transfer_chirality(original_smiles , protonated_smiles , timeout=10) :
    """Transfer chirality with timeout and better error handling."""
    try :
        # Create molecules from SMILES
        mol_orig = Chem.MolFromSmiles(original_smiles)
        mol_prot = Chem.MolFromSmiles(protonated_smiles)

        if mol_orig is None or mol_prot is None :
            return protonated_smiles

        # Check if the original molecule has chiral centers
        chiral = False
        for atom in mol_orig.GetAtoms() :
            if atom.GetChiralTag() != Chem.rdchem.ChiralType.CHI_UNSPECIFIED :
                chiral = True
                break

        if not chiral :
            return protonated_smiles

        # Try substructure matching first
        match = mol_prot.GetSubstructMatch(mol_orig)

        # If substructure matching fails, try MCS with timeout
        if not match :
            mcs = rdFMCS.FindMCS([ mol_orig , mol_prot ] ,
                                 completeRingsOnly=True ,
                                 matchValences=False ,
                                 ringMatchesRingOnly=True ,
                                 matchChiralTag=False ,
                                 timeout=timeout)  # Add timeout

            patt = Chem.MolFromSmarts(mcs.smartsString)
            if patt :
                orig_match = mol_orig.GetSubstructMatch(patt)
                prot_match = mol_prot.GetSubstructMatch(patt)

                if orig_match and prot_match :
                    # Create mapping from original to protonated atoms
                    atom_map = { }
                    for i , j in zip(orig_match , prot_match) :
                        atom_map[ i ] = j

                    # Transfer chirality
                    for orig_idx , orig_atom in enumerate(mol_orig.GetAtoms()) :
                        chiral_tag = orig_atom.GetChiralTag()
                        if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED and orig_idx in atom_map :
                            prot_atom = mol_prot.GetAtomWithIdx(atom_map[ orig_idx ])
                            prot_atom.SetChiralTag(chiral_tag)
                else :
                    return protonated_smiles
            else :
                return protonated_smiles
        else :
            # If substructure matching worked, transfer chirality as before
            for orig_idx , prot_idx in enumerate(match) :
                orig_atom = mol_orig.GetAtomWithIdx(orig_idx)
                prot_atom = mol_prot.GetAtomWithIdx(prot_idx)
                chiral_tag = orig_atom.GetChiralTag()
                if chiral_tag != Chem.rdchem.ChiralType.CHI_UNSPECIFIED :
                    prot_atom.SetChiralTag(chiral_tag)

        # Optionally, you can reassign stereochemistry (which may also update CIP labels).
        Chem.AssignStereochemistry(mol_prot, force=True, cleanIt=True)

        # Return SMILES with chirality information
        return Chem.MolToSmiles(mol_prot , isomericSmiles=True)
    except Exception as e :
        print(f"Error processing {original_smiles}: {str(e)}")
        return protonated_smiles


def process_transfer_chirality_in_batches(df , batch_size=100) :
    """Process the dataframe in batches to avoid memory issues."""
    result_df = df.copy()
    result_df[ 'Predicted pKa smiles updated' ] = result_df[ 'Predicted pKa smiles' ]

    # Only process rows where SMILES are different
    rows_to_process = df[ df[ 'Smiles' ] != df[ 'Predicted pKa smiles' ] ]
    total_rows = len(rows_to_process)

    print(f"Processing {total_rows} molecules with chiral centers in batches of {batch_size}")

    for i in tqdm(range(0 , total_rows , batch_size), desc="Transferring chirality") :
        batch = rows_to_process.iloc[ i :i + batch_size ]

        for _ , row in batch.iterrows() :
            idx = row.name
            orig = row[ 'Smiles' ]
            prot = row[ 'Predicted pKa smiles' ]
            updated = transfer_chirality(orig , prot)
            result_df.at[ idx , 'Predicted pKa smiles updated' ] = updated

    return result_df


# Example usage
if __name__ == '__main__' :
    csv_path = r'C:\Users\Jerome Genzling\OneDrive - McGill University\Documents\Research\pKa predictor\ACIE Submission\pka_predictor_ben\Datasets\onlycsv.csv'
    csv = pd.read_csv(csv_path)

    # Process in batches
    result = process_transfer_chirality_in_batches(csv , batch_size=50)

    # Output results
    result.to_csv(csv_path.replace('.csv' , '_updated.csv') , index=False)