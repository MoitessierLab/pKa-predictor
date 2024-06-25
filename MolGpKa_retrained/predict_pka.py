#!/usr/bin/env python
# coding: utf-8

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from rdkit.Chem.MolStandardize import rdMolStandardize

import os.path as osp
import numpy as np
import pandas as pd
import warnings

import torch
from utils.ionization_group import get_ionization_aid
from utils.descriptor import mol2vec
from utils.net import GCNNet

root = osp.abspath(osp.dirname(__file__))


def load_model(model_file, device="cpu"):
    model= GCNNet().to(device)
    model.load_state_dict(torch.load(model_file, map_location=device))
    model.eval()
    return model

def model_pred(m2, aid, model, device="cpu"):
    data = mol2vec(m2, aid)
    with torch.no_grad():
        data = data.to(device)
        pKa = model(data)
        pKa = pKa.cpu().numpy()
        pka = pKa[0][0]
    return pka

def predict_acid(mol):
    model_file = osp.join(root, "../models/weight_acid.pth")
    model_acid = load_model(model_file)

    acid_idxs= get_ionization_aid(mol, acid_or_base="acid")
    acid_res = {}
    for aid in acid_idxs:
        apka = model_pred(mol, aid, model_acid)
        acid_res.update({aid:apka})
    return acid_res

def predict_base(mol):
    model_file = osp.join(root, "../models/weight_base.pth")
    model_base = load_model(model_file)

    base_idxs= get_ionization_aid(mol, acid_or_base="base")
    base_res = {}
    for aid in base_idxs:
        bpka = model_pred(mol, aid, model_base) 
        base_res.update({aid:bpka})
    return base_res

def predict(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    return base_dict, acid_dict

def predict_for_protonate(mol, uncharged=True):
    if uncharged:
        un = rdMolStandardize.Uncharger()
        mol = un.uncharge(mol)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    mol = AllChem.AddHs(mol)
    base_dict = predict_base(mol)
    acid_dict = predict_acid(mol)
    return base_dict, acid_dict, mol

'''
if __name__=="__main__":
    mol = Chem.MolFromSmiles("CC(C)[NH3+]")
    print('UNCHARGED = TRUE')
    base_dict, acid_dict = predict(mol, uncharged=True)
    print("base:",base_dict)
    print("acid:",acid_dict)
    print('UNCHARGED = FALSE')
    base_dict, acid_dict = predict(mol, uncharged=False)
    print("base:", base_dict)
    print("acid:", acid_dict)
'''

df = pd.read_csv('test_set_noduplicates.csv')
warnings.filterwarnings("ignore")
for index,row in df.iterrows():
    mol = Chem.MolFromSmiles(row['Smiles'])
    mol_H = Chem.AddHs(mol)
    charge = Chem.rdmolops.GetFormalCharge(mol)
    print(row['Smiles'],charge)
    base_dict, acid_dict = predict(mol, uncharged=True)
    df.loc[index, 'MolGpKa_base'] = str(base_dict)
    df.loc[index, 'MolGpKa_acid'] = str(acid_dict)

df.to_csv('MolGpKa_TestPredictions.csv', index=False)

