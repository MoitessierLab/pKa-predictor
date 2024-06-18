# pKa-predictor
Leveraging our Teaching Experience to Improve Machine Learning: Application to pKa Prediction.
Jérôme Genzling, Ziling Luo, Benjamin Weiser, Nicolas Moitessier
nicolas.moitessier@mcgill.ca
2023-12-07 - revised 2024-06-18

# Required libraries:
torch, torch_geometric, pandas, numpy, rdkit, seaborn, hyperopt

# Repository Structure

- The complete assembled and clean data set can be found in the data folder.
- The clustered and randomly splited sets (obtained with the split_train_test_by_TC.py) can be found in the Clusters_Max_TC folder.
- The code to generate the various fingerprints used for the Baseline Models can be found in Baseline_Models/Descriptors.
- These latter will then be used in the respective folders for the traditional models (Baseline_Models/RF or Baseline_Models/XGB).
- All the code related to our GNN/GAT model can be found in the GNN folder.

# Getting started with our GNN model
Command to see the usage of this python script:
python main.py --mode usage

Running any of the mode will first output all the keyword and their default values.

To run the provided model on the csv file called train_set_0.65.csv:
1. Create a folder named pickled_data
2. Run the following command:
python3 main.py --mode test ---n_graph_layers 4 --mask_size 4 --data_path Datasets/ --input train_set_0.65.csv --verbose 2 --output testing_model --n_random_smiles 0 --model_name model_4-4.pth

