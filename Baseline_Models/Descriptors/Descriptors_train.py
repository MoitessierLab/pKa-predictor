import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors
import numpy as np
from rdkit.Chem import MACCSkeys
from rdkit.Chem import AllChem
from rdkit.Chem.EState.Fingerprinter import FingerprintMol

df = pd.read_csv('train_set_0.65.csv')

smiles = df['Smiles']
name = df['Name']
pKas = df['pKa']

rdkit_2d_desc = []

MolFromSmiles = [Chem.MolFromSmiles(i) for i in smiles]

for mol in MolFromSmiles:
   calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
   header = calc.GetDescriptorNames()
   ds = calc.CalcDescriptors(mol)
   rdkit_2d_desc.append(ds)


df = pd.DataFrame(rdkit_2d_desc,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('2D_train_0.65.csv', index=False)

MACCs = []
header = []


for mol in MolFromSmiles:
    MACCs.append(np.array(MACCSkeys.GenMACCSKeys(mol)))

for i in range(len(MACCs[0])):
   header.append('MACC_fp' + str(i+1))

df = pd.DataFrame(MACCs,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('MACCs_train_0.65.csv', index=False)


Morgan = []

header = []

for mol in MolFromSmiles:
   Morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=1024)))

for i in range(len(Morgan[0])):
   header.append('Morgan_fp' + str(i+1))

df = pd.DataFrame(Morgan,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('Morgan_1024_2_train_0.65.csv', index=False)


Morgan = []

header = []

for mol in MolFromSmiles:
   Morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=3, nBits=1024)))

for i in range(len(Morgan[0])):
   header.append('Morgan_fp' + str(i+1))

df = pd.DataFrame(Morgan,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('Morgan_1024_3_train_0.65.csv', index=False)



Morgan = []

header = []
for mol in MolFromSmiles:
       Morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048)))

for i in range(len(Morgan[0])):
   header.append('Morgan_fp' + str(i+1))

df = pd.DataFrame(Morgan,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('Morgan_2048_2_train_0.65.csv', index=False)

#
Morgan = []

header = []
for mol in MolFromSmiles:
       Morgan.append(np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=3, nBits=2048)))

for i in range(len(Morgan[0])):
   header.append('Morgan_fp' + str(i+1))

df = pd.DataFrame(Morgan,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('Morgan_2048_3_train_0.65.csv', index=False)


EState = []

header = []
for mol in MolFromSmiles:
    EState.append(np.array(FingerprintMol(mol)[0]))

for i in range(len(EState[0])):
   header.append('EState_fp' + str(i+1))

df = pd.DataFrame(EState,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('EState_train_0.65.csv', index=False)


EStateSum = []
header = []
for mol in MolFromSmiles:
       EStateSum.append(np.array(FingerprintMol(mol)[1]))

for i in range(len(EStateSum[0])):
   header.append('EState_fp' + str(i+1))

df = pd.DataFrame(EStateSum,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('EStateSum_train_0.65.csv', index=False)

twod_MACCs = []
twod_header = []
maccs_header = []


for mol in MolFromSmiles:
       calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
       twod_header = calc.GetDescriptorNames()
       ds = np.array(calc.CalcDescriptors(mol))
       maccs = np.array(MACCSkeys.GenMACCSKeys(mol))
       all_info = np.concatenate((ds, maccs), axis=None)
       twod_MACCs.append(all_info)

for i in range(len(maccs)):
   maccs_header.append('MACC_fp' + str(i+1))

header = np.concatenate((twod_header, maccs_header), axis=None)
df = pd.DataFrame(twod_MACCs,columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('2D_MACCs_train_0.65.csv', index=False)

twod_Morgan = []
twod_header = []
Morgan_header = []

for mol in MolFromSmiles:
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        twod_header = calc.GetDescriptorNames()
        ds = np.array(calc.CalcDescriptors(mol))
        Morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits=2048))
        all_info = np.concatenate((ds, Morgan), axis=None)
        twod_Morgan.append(all_info)

for i in range(len(Morgan)):
    Morgan_header.append('Morgan_fp' + str(i+1))

header = np.concatenate((twod_header, Morgan_header), axis=None)
df = pd.DataFrame(twod_Morgan, columns=header)
df.insert(loc=0, column='name', value=name)
df.insert(loc=1, column='pKa', value=pKas)
df.to_csv('2D_Morgan_train_0.65.csv', index=False)
