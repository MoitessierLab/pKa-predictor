# pKa and protonation state predictor
# Jerome Genzling, Ziling Luo, Ben Weiser, Nicolas Moitessier
# Department of Chemistry, McGill University
# Montreal, QC, Canada
#
# Some preliminary installation:
# sudo apt install python3-pip
# pip install torch
# pip install numpy
# pip install torch_geometric
# pip install pandas
# pip install rdkit
# pip install seaborn
# pip install hyperopt
# pip install scikit-learn
# To visualise smiles, you may use: https://www.cheminfo.org/flavor/malaria/Utilities/SMILES_generator___checker/index.html

import torch
import numpy as np
import random
import os
import time

from torch_geometric.loader import DataLoader

from hyperoptimize import hyperoptimize
from plot_and_print import print_model_txt
from prepare_set import generate_datasets, dump_datasets
from argParser import argsParser
from utils import set_cuda_visible_device, load_data
from train_pKa_predictor import training, testing, inferring, testing_with_IC
from usage import usage


if __name__ == '__main__':

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| %s                                                                                                        |' % s)

    args = argsParser()
    
    seed = 42
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    np.random.seed(args.seed)
    random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.mode == 'train':
        print('| Checking if GPU is available                                                                                               |')
        gpu = torch.cuda.is_available()
        if gpu:
            gpu_name = torch.cuda.get_device_name(0)
            print('| %20s is available                                                                                          |' % gpu_name)
        else:
            print('| No GPU available                                                                                                           |')
        print('|----------------------------------------------------------------------------------------------------------------------------|')
    
        set_cuda_visible_device(args.ngpu)
    
        if gpu:
            cmd = set_cuda_visible_device(0)
            os.environ['CUDA_VISIBLE_DEVICES'] = cmd[:-1]
    
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Training the model
    if args.mode == "train":
        train_file = args.data_path + args.train_data
        train_dataset = generate_datasets(train_file, 'Train', args)
        train_path = args.train_pickled
        dump_datasets(train_dataset, train_path)

        if args.test_data != "none":
            test_file = args.data_path + args.test_data
            test_dataset = generate_datasets(test_file, 'Test', args)
            test_path = args.test_pickled
            dump_datasets(test_dataset, test_path)

        # Loading data for training (if no testing set provided, use the training set as testing set).
        train_data = load_data(args.train_pickled)
        if args.test_data != "none":
            test_data = load_data(args.test_pickled)
        else:
            test_data = load_data(args.train_pickled)

        best_hypers = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler_gamma': args.scheduler_gamma,
            'model_embedding_size': args.embedding_size,
            'model_gnn_layers': args.n_graph_layers,
            'model_fc_layers': args.n_FC_layers,
            'model_dropout_rate': args.dropout_rate,
            'model_dense_neurons': args.model_dense_neurons,
            'model_attention_heads': args.model_attention_heads,
        }
        print('| Preparing the data loaders...                                                                                              |', flush=True)
        train_loader = DataLoader(train_data, best_hypers["batch_size"],
                                  num_workers=args.num_workers, shuffle=True)

        test_loader = DataLoader(test_data, best_hypers["batch_size"],
                                 num_workers=args.num_workers, shuffle=False)

        print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
        # training the model
        best_trained_model = training(train_dataset, best_hypers, train_loader, test_loader, args)
        
        torch.save(best_trained_model.state_dict(), args.save_dir + args.output + '.pth')

        testing(best_trained_model, train_loader, test_loader, args)

    # Optimizing the hyperparameters
    if args.mode == "hyperopt":
        train_file = args.data_path + args.train_data
        train_dataset = generate_datasets(train_file, 'Train', args)
        train_path = args.train_pickled
        dump_datasets(train_dataset, train_path)

        if args.test_data != "none":
            test_file = args.data_path + args.test_data
            test_dataset = generate_datasets(test_file, 'Test', args)
            test_path = args.test_pickled
            dump_datasets(test_dataset, test_path)

        # Loading data for training (if no testing set provided, use the training set as testing set).
        train_data = load_data(args.train_pickled)
        if args.test_data != "none":
            test_data = load_data(args.test_pickled)
        else:
            test_data = load_data(args.train_pickled)

        print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
        # training the model
        hyperoptimize(train_dataset, train_data, test_data, args)

    elif args.mode == 'usage':
        usage()

    elif args.mode == 'write_model':
        hypers = {
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'weight_decay': args.weight_decay,
            'scheduler_gamma': args.scheduler_gamma,
            'model_embedding_size': args.embedding_size,
            'model_gnn_layers': args.n_graph_layers,
            'model_fc_layers': args.n_FC_layers,
            'model_dropout_rate': args.dropout_rate,
            'model_dense_neurons': args.model_dense_neurons,
            'model_attention_heads': args.model_attention_heads,
            'model_node_feature_size': args.node_feature_size,
            'model_edge_feature_size': args.edge_feature_size,
        }

        print_model_txt(hypers, args)

    elif args.mode == 'test_with_IC':
        testing_with_IC(args)

    else:
        inferring(args)

    now = time.localtime()
    s = "%04d-%02d-%02d %02d:%02d:%02d" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Job finished at %s                                                                                        |' % s)
    print('|----------------------------------------------------------------------------------------------------------------------------|')
