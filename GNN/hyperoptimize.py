import copy

import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

from utils import load_data, calculate_metrics, average, optimizer_to
from plot_and_print import plot_figure1, plot_figure2, plot_figure3, print_results, print_inference, print_results_test
from prepare_set import generate_infersets, dump_datasets
from GNN import GNN
from train import train, evaluate
from torch_geometric.loader import DataLoader


def hyperoptimize(train_dataset_0, train_data_0, test_data_0, args_0):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def training_hyperopt(train_dataset, hypers, train_data, test_data, args):
        print('| Preparing the data loaders...                                                                                              |', flush=True)
        train_loader = DataLoader(train_data, hypers["batch_size"],
                                  num_workers=args.num_workers, shuffle=True)

        test_loader = DataLoader(test_data, hypers["batch_size"],
                                 num_workers=args.num_workers, shuffle=False)

        model_params = {k: v for k, v in hypers.items() if k.startswith("model_")}

        trained_model = GNN(feature_size=train_dataset[0].x.shape[1],
                            edge_dim=train_dataset[0].edge_attr.shape[1],
                            model_params=model_params)

        loss_fn = torch.nn.MSELoss()

        optimizer = torch.optim.Adam(trained_model.parameters(),
                                     lr=hypers["learning_rate"],
                                     weight_decay=hypers["weight_decay"])

        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=hypers["scheduler_gamma"])

        trained_model.to(device)
        train_loss_all = []
        test_loss_all = []

        print('| Now training the model...                                                                                                  |')
        print('|----------------------------------------------------------------------------------------------------------------------------|')
        print('| learning_rate:          %-98s |' % hypers['learning_rate'])
        print('| weight_decay:           %-98s |' % hypers['weight_decay'])
        print('| scheduler_gamma:        %-98s |' % hypers['scheduler_gamma'])
        print('| model_embedding_size:   %-98s |' % hypers['model_embedding_size'])
        #print('| model_gnn_layers:       %-98s |' % hypers['model_gnn_layers'])
        #print('| model_fc_layers:        %-98s |' % hypers['model_fc_layers'])
        print('| model_dropout_rate:     %-98s |' % hypers['model_dropout_rate'])
        print('| model_dense_neurons:    %-98s |' % hypers['model_dense_neurons'])
        print('| model_attention_heads:  %-98s |' % hypers['model_attention_heads'])
        print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

        start_time = time.time()

        best_val_loss = 250
        same_val_loss = 0

        for epoch in range(1, args.epoch + 1):
            # Training
            trained_model.train()
            train_loss = train(epoch, trained_model, train_loader, optimizer, loss_fn, args)
            train_loss_all.append(train_loss)

            # Validation
            trained_model.eval()
            val_loss = evaluate(epoch, trained_model, test_loader, loss_fn, args)
            test_loss_all.append(val_loss)
            now = time.time()
            print('| epoch # %4s | train loss %6.3f       | test loss %6.3f        | time (min): %8.2f  | time (hours): %8.2f          |               '
                  % (epoch, train_loss, val_loss, (now - start_time) / 60.0, (now - start_time) / 3600.0), flush=True)

            scheduler.step()

            # save the model is better on the validation set or close to the best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                same_val_loss = 0
                torch.save(trained_model.state_dict(), args.save_dir + args.output + '_' + str(epoch) + '.pth')
            elif val_loss < best_val_loss + 0.05:
                torch.save(trained_model.state_dict(), args.save_dir + args.output + '_' + str(epoch) + '.pth')
                same_val_loss = 0

            if val_loss > best_val_loss + 0.05:
                same_val_loss = same_val_loss + 1
                if same_val_loss >= args.hyperopt_convergence:
                    break
            if val_loss > best_val_loss + args.hyperopt_max_increase and epoch > 100:
                break

            # overwrite the "last" model
            torch.save({
                'epoch': epoch,
                'model_state_dict': trained_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_all,
                'test_loss': test_loss_all,
            }, args.save_dir + args.output + '_last.pth')

        print('|----------------------------------------------------------------------------------------------------------------------------|')
        print('| best val loss:          %-98s |' % best_val_loss)
        print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

        return best_val_loss

    hypers_0 = {
        'batch_size': hp.choice('batch_size', [args_0.batch_size]),  # batch size defined when creating the data
        'learning_rate': hp.choice('learning_rate', [0.002, 0.001, 0.0005, 0.0002]),
        'weight_decay': hp.choice('weight_decay', [0.000002, 0.000001, 0.0000005]),
        'scheduler_gamma': hp.choice('scheduler_gamma', [0.99, 0.985, 0.98, 0.97]),
        'model_embedding_size': hp.choice('model_embedding_size', [64, 96, 128, 160]),
        'model_gnn_layers': args_0.n_graph_layers,
        'model_fc_layers': args_0.n_FC_layers,
        'model_dropout_rate': hp.choice('model_dropout_rate', [0.1, 0.15, 0.20, 0.25]),
        'model_dense_neurons': hp.choice('model_dense_neurons', [196, 256, 320, 384, 448]),
        'model_attention_heads': hp.choice('model_attention_heads', [3, 4]),
    }

    best_score = 1000
    best_params = None

    def f(params):
        nonlocal best_score
        nonlocal best_params
        acc = training_hyperopt(train_dataset_0, params, train_data_0, test_data_0, args_0)
        if acc < best_score:
            best_score = acc
            best_params = params
        return {'loss': -acc, 'status': STATUS_OK}

    trials = Trials()

    best_hypers = fmin(
        fn=f,
        space=hypers_0,
        algo=tpe.suggest,
        max_evals=args_0.hyperopt_max_evals,
        trials=trials
    )

    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| Final best parameters:                                                                                                     |')
    print('| learning_rate:          %98.6f |' % best_params['learning_rate'])
    print('| weight_decay:           %98.6f |' % best_params['weight_decay'])
    print('| scheduler_gamma:        %98.4f |' % best_params['scheduler_gamma'])
    print('| model_embedding_size:   %98.0f |' % best_params['model_embedding_size'])
    print('| model_dropout_rate:     %98.2f |' % best_params['model_dropout_rate'])
    print('| model_dense_neurons:    %98.0f |' % best_params['model_dense_neurons'])
    print('| model_attention_heads:  %98.0f |' % best_params['model_attention_heads'])
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
