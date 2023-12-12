
def usage():
    print('| Creating the testing and training set with maximum dissimilarities (the set will be in a folder called Clusters_Max_TC):   |')
    print('|    main.py --split_train_test_by_TC.py                                                                                     |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| For training a model and optimizing hyperparameters using train_set.csv and testing on val_set.csv:                        |')
    print('|    python3 main.py --mode hyperopt --n_graph_layers 4 --mask_size 4 --data_path data/ --train_data train_set.csv           |')
    print('|    --test_data test_set.csv --verbose 2 --output hyper_optimize --n_random_smiles 100 --ngpu 2 --epoch 1000                |')
    print('|          --train_pickled pickled_data/train_pickled_4-4.pkl --test_pickled pickled_data/val_pickled_4-4.pkl                |')'
    print('|    --test_source_2 "RBF" --test_data dataset.csv --n_graph_layers 3 --mask_size 4  --verbose 2 --output train_output       |')
    print('|    --n_random_smiles 50 --ngpu 1 --epoch 500                                                                               |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| For training a model using only the data from the Moitessier_Set and testing on the data from the RBF set (Moitessier_set  |')
    print('| and RBF are labels in the dataset.csv                                                                                      |')
    print('|    python3 main.py --mode train --lr 0.0005 --train_source_1 "Moitessier_Set" --train_data dataset.csv                     |')
    print('|    --test_source_2 "RBF" --test_data dataset.csv --n_graph_layers 3 --mask_size 4  --verbose 2 --output train_output       |')
    print('|    --n_random_smiles 50 --ngpu 1 --epoch 500                                                                               |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| For testing the model on a set (observed pKa must be provided) - multiple protonation states will be provided:             |')
    print('|    main.py --mode test --input test.csv --model_dir saved_models/ --model_name train_output.pth --n_graph_layers 3         |')
    print('|    --mask_size 4                                                                                                           |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| For predicting pKa using a model on a set:                                                                                 |')
    print('|    python3 main.py --mode infer --input data.csv --model_dir saved_models/ --model_name train_output.pth                   |')
    print('|    --n_graph_layers 3 --mask_size 4 >                                                                                      |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| For predicting the most likely protonation state at a given pH:                                                            |')
    print('|    python3 main.py --mode pH --pH 7.4 --input data.csv --model_dir saved_models/ --model_name train_output.pth             |')
    print('|    --n_graph_layers 3 --mask_size 4 >                                                                                      |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')

