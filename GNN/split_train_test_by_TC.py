import copy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
keras.backend.clear_session()
import random

global id_in_test_train
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from tqdm import tqdm

# Set seed
seed = 1
os.environ['PYTHONHASHEDSEED'] = str(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

print('|-----------------------------------------------------------------------|')
print('| Clustering by similarity and preparing training and testing sets.     |')
print('| Code by Ben Weiser and Nic Moitessier                                 |')
print('| Department of Chemistry, McGill University                            |')
print('| Montreal, QC, Canada                                                  |')
print('| nicolas.moitessier@mcgill.ca                                          |')
print('|                                                                       |')
print('| 2023-08-29                                                            |')
print('|-----------------------------------------------------------------------|')


def load_parameters():
    # Parameters set here.
    pm = {'dir': '/home/moitessi/pKaPredictor2',
          'data_dir': '/Clusters_Max_TC/',
          'fig_dir': 'Figures',
          'model_dir': 'Models',
          'num_test_set_clusters': 250,
          'test_set_cluster_size': 10,
         } 
    return pm


tanimoto = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


# Make a dataframe with columns for each Tanimoto coefficient value
pm = load_parameters()
DATA_DIR = pm['data_dir']
dir_path = pm['dir'] + DATA_DIR

# Make directory for Tanimoto smiles csv in pm['dir']
if not os.path.exists(pm['dir'] + '/Clusters_Max_TC'):
    os.makedirs(pm['dir'] + '/Clusters_Max_TC')
    
tanimoto_df = pd.DataFrame(columns=tanimoto)
docked = []

# loop over files in directory and subdirectories
allsmiles = pd.read_csv(dir_path + '/full_set.csv')

print('| Smiles loaded                                                         |')

Novartis_test_set = True
Literature_test_set = True
# Make Test Set
##############################################################################################################
if Novartis_test_set is False and Literature_test_set is False:
    test_set = pd.DataFrame(columns=['Smiles', 'tanimoto_coefficient'])
    for i in tqdm(range(pm['num_test_set_clusters']), total=pm['num_test_set_clusters'],
                                                      desc="| Preparing Test Set                                                    |\n"):
        origin_seed = random.randint(0, len(allsmiles))-1
        test_set_origin = allsmiles['Smiles'].iloc[origin_seed]  # pick a random id from docked_ids to be the test set origin

        # Get the Tanimoto similarity between the test set origin and all smiles

        # keep these 500 and call them the test ids
        origin_mol = AllChem.MolFromSmiles(test_set_origin)
        origin_bit = AllChem.GetMorganFingerprintAsBitVect(origin_mol, radius=2, nBits=2048)
        origin_tanimoto_coefficient = pd.DataFrame(columns=['Smiles', 'tanimoto_coefficient'])

        for test_compound in allsmiles['Smiles']:
            test_mol = Chem.MolFromSmiles(test_compound)
            test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
            tanimoto_coefficient = DataStructs.TanimotoSimilarity(origin_bit, test_bit)
            origin_tanimoto_coefficient = pd.concat([origin_tanimoto_coefficient, pd.DataFrame({'Smiles': [test_compound],
                                                                                                'tanimoto_coefficient': [tanimoto_coefficient]})])

        # Add id column of allsmiles to origin_tanimoto_coefficient and merge on smiles
        test_set_cluster = origin_tanimoto_coefficient.merge(allsmiles, on='Smiles')
        # Sort origin_tanimoto_coefficient by tanimoto_coefficient and take top 500
        test_set_cluster = test_set_cluster.sort_values(by=['tanimoto_coefficient'], ascending=False)
        test_set_cluster = test_set_cluster.head(pm['test_set_cluster_size'])
        test_set = pd.concat([test_set, test_set_cluster])
    # Get size for test set dataframe
    print('Size of test set:', len(test_set))
    test_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'test_set.csv', index=False)
elif Novartis_test_set is True:
    testsmiles = pd.read_csv(dir_path + '/test_set_Novartis.csv')
    test_set = pd.DataFrame(columns=['Name', 'pKa', 'Center', 'Index', 'Smiles'])
    for index, row in testsmiles.iterrows():
        test_set = pd.concat([test_set, pd.DataFrame([row], columns=row.index)])
    # Get size for test set dataframe
    #print('Size of test set:', len(test_set))
    #test_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'test_set_Novartis.csv', index=False)
elif Literature_test_set is True:
    testsmiles = pd.read_csv(dir_path + '/test_set_Literature.csv')
    test_set = pd.DataFrame(columns=['Name', 'pKa', 'Center', 'Index', 'Smiles'])
    for index, row in testsmiles.iterrows():
        test_set = pd.concat([test_set, pd.DataFrame([row], columns=row.index)])
    # Get size for test set dataframe
    #print('Size of test set:', len(test_set))
    #test_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'test_set_Novartis.csv', index=False)

# Make Train Set
##############################################################################################################

print('Making Train Set')
# Drop test set from Allsmiles
train_smiles = copy.deepcopy(allsmiles)
# Drop test set from train_smiles
train_smiles = train_smiles[~train_smiles['Smiles'].isin(test_set['Smiles'])]

most_similar_to_in_train = pd.DataFrame(columns=['Smiles', 'tanimoto_coefficient'])
for train_compound in tqdm(train_smiles['Smiles'], total=len(train_smiles), 
                                                   desc="| Preparing Training Set                                                |\n"):
    train_mol = Chem.MolFromSmiles(train_compound)
    train_bit = AllChem.GetMorganFingerprintAsBitVect(train_mol, radius=2, nBits=2048)
    train_similarities = pd.DataFrame(columns=['tanimoto_coefficient'])
    for test_compound in test_set['Smiles']:
        # Convert the train compound to a mol object
        test_mol = Chem.MolFromSmiles(test_compound)
        test_bit = AllChem.GetMorganFingerprintAsBitVect(test_mol, radius=2, nBits=2048)
        tanimoto_coefficient = DataStructs.TanimotoSimilarity(test_bit, train_bit)
        train_similarities = pd.concat([train_similarities, pd.DataFrame({'tanimoto_coefficient': [tanimoto_coefficient]})])
    
    most_similar_to_in_train = pd.concat([most_similar_to_in_train, pd.DataFrame({'Smiles': [train_compound], 'tanimoto_coefficient': [train_similarities['tanimoto_coefficient'].max()]})])
 
for MAX_TANIMOTO in tanimoto:
    # Keep only rows of most_similar_to_in_train where tanimoto_coefficient is less than MAX_TANIMOTO
    most_similar_to_in_train_tan = most_similar_to_in_train[most_similar_to_in_train['tanimoto_coefficient'] < MAX_TANIMOTO]
    train_set = most_similar_to_in_train_tan.merge(allsmiles, on='Smiles')
    # Get size for train set dataframe
    print('| Maximum Tanimoto coefficient: %5.3f, Size of the training set: %6.0f |' % (MAX_TANIMOTO, len(train_set)))
    # Put size of train_set into row 'train size and tanimoto column into tanimoto_df for plotting
    tanimoto_df.loc['train_set_size', MAX_TANIMOTO] = len(train_set)
    if Novartis_test_set is False and Literature_test_set is False:
        train_set.to_csv(pm['dir']+'/Clusters_Max_TC/' + 'train_set_' + str(MAX_TANIMOTO) + '.csv', index=False)
        tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df.csv', index=False)
    elif Novartis_test_set is True:
        train_set.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_fromNS_' + str(MAX_TANIMOTO) + '.csv', index=False)
        tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df_fromNS.csv', index=False)
    elif Literature_test_set is True:
        train_set.to_csv(pm['dir']+'/Clusters_Max_TC/' + 'train_set_fromLitt_' + str(MAX_TANIMOTO) + '.csv', index=False)
        tanimoto_df.to_csv(pm['dir'] + '/Clusters_Max_TC/' + 'tanimoto_df_fromLitt.csv', index=False)

# Plot using matplot lib train_set_size over tanimoto
plt.plot(tanimoto_df.columns, tanimoto_df.loc['train_set_size'])
plt.xlabel('Tanimoto Coefficient')
plt.ylabel('Train Set Size')
plt.title('Train Set Size vs Tanimoto Coefficient')

# Add y axis value to each point
for i in tanimoto:
    plt.annotate(tanimoto_df.loc['train_set_size'][i], (i,tanimoto_df.loc['train_set_size'][i]))
plt.savefig(pm['dir'] + '/Clusters_Max_TC/' + 'train_set_size_vs_tanimoto.png')
plt.close()

print('|-----------------------------------------------------------------------|')
print('| Calculations complete                                                 |')
print('|-----------------------------------------------------------------------|')

