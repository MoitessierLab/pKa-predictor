# pKa-predictor
Leveraging our Teacher’s Experience to Improve Machine Learning: Application to pKa Prediction.
Jérôme Genzling, Ziling Luo, Benjamin Weiser, Nicolas Moitessier
nicolas.moitessier@mcgill.ca
2023-12-07

# Required libraries:
torch, torch_geometric, pandas, numpy, rdkit, seaborn, hyperopt

# Repository Structure

The complete assembled and clean data set can be found in the data folder.
The clustered and randomly splited sets (obtained with the split_train_test_by_TC.py) can be found in the Clusters_Max_TC folder.
The code to generate the various fingerprints used for the Baseline Models can be found in Baseline_Models/Descriptors.
These latter will then be used in the respective folders for the traditional models (Baseline_Models/RF or Baseline_Models/XGB).
All the code related to our GNN/GAT model can be found in the GNN folder.

# Getting started with our GNN model
Command to see the usage of this python script:
python main.py --mode usage

Running any of the mode will first output all the keyword and their default values.
