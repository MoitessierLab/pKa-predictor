import copy

import torch
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from utils import load_data, calculate_metrics, average, optimizer_to
from plot_and_print import plot_figure1, plot_figure2, plot_figure3, print_results, print_inference, print_results_test
from prepare_set import generate_infersets, dump_datasets, generate_datasets
from GNN import GNN, GNN_New
from train import train, evaluate
from torch_geometric.loader import DataLoader


def training(train_dataset, best_hypers, train_loader, test_loader, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_params = {k: v for k, v in best_hypers.items() if k.startswith("model_")}

    if args.GATv2Conv_Or_Other == "GATv2Conv":
        best_trained_model = GNN(feature_size=train_dataset[0].x.shape[1],
                                 edge_dim=train_dataset[0].edge_attr.shape[1],
                                 model_params=model_params)
    else:
        best_trained_model = GNN_New(feature_size=train_dataset[0].x.shape[1],
                                     edge_dim=train_dataset[0].edge_attr.shape[1],
                                     model_params=model_params)

    loss_fn = torch.nn.MSELoss()

    optimizer = torch.optim.Adam(best_trained_model.parameters(),
                                 lr=best_hypers["learning_rate"],
                                 weight_decay=best_hypers["weight_decay"])

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       gamma=best_hypers["scheduler_gamma"])

    first_epoch = 1
    if args.restart != 'none':
        print('| Loading previously trained model...                                                                                        |')
        checkpoint = torch.load(args.model_dir + args.restart)
        best_trained_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        optimizer_to(optimizer, device)
        first_epoch = checkpoint['epoch'] + 1

    best_trained_model.to(device)
    train_loss_all = []
    test_loss_all = []

    if args.restart != 'none':
        train_loss_all = checkpoint['train_loss']
        test_loss_all = checkpoint['test_loss']

    print('| Now training the model...                                                                                                  |')
    print('|----------------------------------------------------------------------------------------------------------------------------|',
          flush=True)

    start_time = time.time()

    best_val_loss = 250

    for epoch in range(first_epoch, args.epoch + 1):
        # Training
        best_trained_model.train()
        train_loss = train(epoch, best_trained_model, train_loader, optimizer, loss_fn, args)
        train_loss_all.append(train_loss)
    
        # Validation
        best_trained_model.eval()
        val_loss = evaluate(epoch, best_trained_model, test_loader, loss_fn, args)
        test_loss_all.append(val_loss)
        now = time.time()
        print('| epoch # %4s | train loss %6.3f       | test loss %6.3f        | time (min): %8.2f  | time (hours): %8.2f          |               '
              % (epoch, train_loss, val_loss, (now-start_time)/60.0, (now-start_time)/3600.0), flush=True)
    
        scheduler.step()

        # save the model is better on the validation set or close to the best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_trained_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_all,
                'test_loss': test_loss_all,
            }, args.save_dir + args.output + '_' + str(epoch) + '.pth')
        elif val_loss < best_val_loss + 0.025:
            torch.save({
                'epoch': epoch,
                'model_state_dict': best_trained_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss_all,
                'test_loss': test_loss_all,
            }, args.save_dir + args.output + '_' + str(epoch) + '.pth')

        # overwrite the "last" model
        torch.save({
            'epoch': epoch,
            'model_state_dict': best_trained_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss_all,
            'test_loss': test_loss_all,
            }, args.save_dir + args.output + '_last.pth')

    print('|----------------------------------------------------------------------------------------------------------------------------|')
    
    plot_figure1(train_loss_all, test_loss_all, args)

    return best_trained_model

    
def final_test(loader, model, loss_fn, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.mode == 'train' else "cpu")

    # Enumerate over the data
    all_preds = []
    all_labels = []
    running_loss = 0.0
    step = 0
    all_centers = []
    all_proposed_centers = []
    all_smiles = []
    all_smiles_base = []
    all_mol_num = []
    all_neutral = []
    all_error = []
    all_ionization_states = []
    model.p = 0
    for batch in loader:
        batch.to(device)

        with torch.no_grad():  # turns of the autograd engine. Less memory and faster

            pred = model(batch.x.float(),
                         batch.edge_index,
                         batch.edge_attr.float(),
                         batch.node_index,
                         batch.mol_formal_charge,
                         batch.center_formal_charge,
                         batch.batch)

        if pred.size()[0] > 1:
            loss = loss_fn(torch.squeeze(pred), batch.y.float())
            all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        else:
            loss = loss_fn(pred[0], batch.y.float())
            all_preds.append(pred[0].cpu().detach().numpy())

        running_loss += loss.item()
        step += 1
        all_smiles.append(batch.smiles)
        all_smiles_base.append(batch.smiles_base)
        all_neutral.append(batch.neutral.cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())
        all_centers.append(batch.ionization_center.cpu().detach().numpy())
        all_proposed_centers.append(batch.proposed_center.cpu().detach().numpy())
        all_mol_num.append(batch.mol_number.cpu().detach().numpy())
        all_error.append(batch.error.cpu().detach().numpy())
        all_ionization_states.append(batch.ionization_state)

    if len(all_preds) > 0 and all_preds[0].size > 1:
        all_preds = np.concatenate(all_preds).ravel()
        all_labels = np.concatenate(all_labels).ravel()
        all_centers = np.concatenate(all_centers).ravel()
        all_proposed_centers = np.concatenate(all_proposed_centers).ravel()
        all_smiles = np.concatenate(all_smiles).ravel()
        all_smiles_base = np.concatenate(all_smiles_base).ravel()
        all_neutral = np.concatenate(all_neutral).ravel()
        all_mol_num = np.concatenate(all_mol_num).ravel()
        all_error = np.concatenate(all_error).ravel()
    elif len(all_preds) > 0:
        all_preds = np.array(all_preds[0]).ravel()
        all_labels = np.array(all_labels[0]).ravel()
        all_centers = np.array(all_centers[0]).ravel()
        all_proposed_centers = np.array(all_proposed_centers[0]).ravel()
        all_smiles = np.array(all_smiles[0]).ravel()
        all_smiles_base = np.array(all_smiles_base[0]).ravel()
        all_neutral = np.array(all_neutral[0]).ravel()
        all_mol_num = np.array(all_mol_num[0]).ravel()
        all_error = np.array(all_error[0]).ravel()

    if step == 0:
        step = 1

    if args.mode == 'train':
        calculate_metrics(all_preds, all_labels, all_mol_num, args)

    return running_loss / step, all_preds, all_labels, all_smiles, all_smiles_base, all_centers, all_proposed_centers,\
        all_mol_num, all_neutral, all_error, all_ionization_states


def testing(best_trained_model, train_loader, test_loader, args):
    loss_fn = torch.nn.MSELoss()
    whiteSpace = ' '
    print('| Training Set %-104s      |' % whiteSpace)
    best_trained_model.eval()

    train_loss, train_predicts, train_labels, train_smiles, train_smiles_base, train_centers, train_proposed_centers, \
        train_mol_num, train_neutral, train_error, train_ionization_states = \
        final_test(model=best_trained_model, loader=train_loader, loss_fn=loss_fn, args=args)

    print('| train loss %-106.3f      |' % train_loss)
    print('|--------------------------------- %-84s      |' % whiteSpace, flush=True)

    print('| Testing Set %-105s      |' % whiteSpace)
    best_trained_model.eval()

    test_loss, test_predicts, test_labels, test_smiles, test_smiles_base, test_centers, test_proposed_centers, \
        test_mol_num, test_neutral, test_error, test_ionization_states = \
        final_test(model=best_trained_model, loader=test_loader, loss_fn=loss_fn, args=args)

    print('| test loss %-106.3f       |' % test_loss)
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

    plot_figure2(train_labels, train_predicts, test_labels, test_predicts, args)

    # average over the random_smiles
    train_predicts, train_labels, train_smiles, train_mol_num, train_error, = \
        average(train_predicts, train_labels, train_smiles, train_mol_num, train_error, args)
    test_predicts, test_labels, test_smiles, test_mol_num, test_error = \
        average(test_predicts, test_labels, test_smiles, test_mol_num, test_error, args)

    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
    print('| Training Set averaged over random smiles %-76s      |' % whiteSpace)
    calculate_metrics(train_predicts, train_labels, train_mol_num, args)
    print('|---------------------------------                                                                                           |', flush=True)
    print('| Testing Set averaged over random smiles  %-76s      |' % whiteSpace)
    calculate_metrics(test_predicts, test_labels, test_mol_num, args)
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)

    print_results(train_predicts, train_labels, train_smiles, train_mol_num, train_error,
                  test_predicts, test_labels, test_smiles, test_mol_num, test_error, args)

    plot_figure3(train_labels, train_predicts, test_labels, test_predicts, args)


def inferring(args):

    infer_file = args.data_path + args.input
    data = pd.read_csv(infer_file, sep=',')
    infer_path = args.infer_pickled

    device = torch.device("cpu")

    best_hypers = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler_gamma': args.scheduler_gamma,
        'model_embedding_size': args.embedding_size,
        'model_gnn_layers': args.n_graph_layers,
        'model_fc_layers': args.n_FC_layers,
        'model_dropout_rate': 0,
        'model_dense_neurons': args.model_dense_neurons,
        'model_attention_heads': args.model_attention_heads,
    }

    model_params = {k: v for k, v in best_hypers.items() if k.startswith("model_")}
    loss_fn = torch.nn.MSELoss()

    print('| Reading the files, preparing the features and computing pKa                                                                |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    if args.mode == 'test':
        print('| mol #  | SMILES                                                                            | center        |      pKa      |')
        print('|        |                                                                                   |  obs. | pred. |  obs. | pred. |')
    else:
        print('| mol #  | SMILES                                                                                   | center | predicted pKa |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')

    library_infer_predicts = []
    library_infer_labels = []
    library_infer_smiles = []
    library_infer_mol_num = []
    library_infer_centers = []
    library_infer_proposed_centers = []
    library_infer_ionization_states = []

    for i, small_mol in tqdm(data.iterrows(), total=data.shape[0]):
        if args.verbose > 1:
            print("|        | Initial inference                                                                        "
                  "                         |")
        if 'Index' in small_mol.keys(): #To resolve not having Index column in the dataset
            initial_proposed_center = int(small_mol['Index']) + 1
        else:
            initial_proposed_center = 0
        ionized_smiles = ''
        initial = True
        infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
            infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles = \
            infer(i, small_mol, initial, ionized_smiles, [], infer_path, model_params, device, best_hypers, loss_fn, args)

        ionized_mol_num = i + 1

        if args.verbose > 1:
            print_inference(infer_predicts, infer_labels, infer_smiles, ionized_smiles, infer_mol_num, ionized_mol_num, infer_centers,
                            initial_proposed_center, initial, args)

        all_infer_predicts = []
        all_infer_labels = []
        all_infer_smiles = []
        all_infer_smiles_base = []
        all_neutral = []
        all_infer_mol_num = []
        all_infer_centers = []
        all_infer_proposed_centers = []
        all_infer_ionization_states = []

        found_pKas = 2
        if len(infer_ionization_states) == 0:
            found_pKas = 0
        else:
            if len(infer_ionization_states[0][0][0]) == 0:
                found_pKas = 1
        # TODO: if none is found, output something (ex.: train_pKas.csv #117)

        # If there is only 1 molecule, it can only be the unchanged one
        # If we have 2 molecules, it is the unchanged + 1 site ionized. No other choice.
        # we can also have the case where we start from N- then get NH, we need another round to get NH2+
        # so we check if we still have a basic nitrogen
        if len(infer_smiles) == 1 and found_pKas < 2:
            all_infer_predicts = copy.deepcopy(infer_predicts)
            all_infer_labels = copy.deepcopy(infer_labels)
            all_infer_smiles = copy.deepcopy(infer_smiles)
            all_infer_mol_num = copy.deepcopy(infer_mol_num)
            all_infer_centers = copy.deepcopy(infer_centers)
            all_infer_proposed_centers = copy.deepcopy(infer_proposed_centers)
            all_infer_ionization_states = copy.deepcopy(infer_ionization_states)

        # in case we have more than 1 ionization center, we look for the one with the highest pKa (max_pka)
        if len(infer_smiles) > 1 or found_pKas == 2:
            max_pKa = 0
            protonation_step = 1

            # We now ionize all the sites one by one in the order of pKa's
            while max_pKa > -10:

                protonation_step += 1
                max_pKa = -11.0
                max_pKa_neutral = -11.0
                if len(infer_predicts) == 0:
                    break

                max_pKa_mol = -1
                max_pKa_mol_neutral = -1
                for j in range(len(infer_predicts)):
                    if infer_neutral[j]:
                        if infer_predicts[j] >= max_pKa_neutral:
                            max_pKa_neutral = infer_predicts[j]
                            max_pKa_mol_neutral = j

                    if infer_predicts[j] >= max_pKa:
                        max_pKa = infer_predicts[j]
                        max_pKa_mol = j

                if max_pKa_mol_neutral != -1 and max_pKa_mol == -1:
                    all_infer_predicts.append(infer_predicts[max_pKa_mol_neutral])
                    all_infer_labels.append(infer_labels[max_pKa_mol_neutral])
                    all_infer_smiles.append(infer_smiles[max_pKa_mol_neutral])
                    all_infer_smiles_base.append(infer_smiles_base[max_pKa_mol_neutral])
                    all_neutral.append(infer_neutral[max_pKa_mol_neutral])
                    all_infer_mol_num.append(infer_mol_num[max_pKa_mol_neutral])
                    all_infer_centers.append(infer_centers[max_pKa_mol_neutral])
                    all_infer_proposed_centers.append(infer_proposed_centers[max_pKa_mol_neutral])
                    all_infer_ionization_states.append(infer_ionization_states[0][max_pKa_mol_neutral])

                if max_pKa_mol != -1:
                    if 'Index' in small_mol.keys():
                        if small_mol['Smiles'] == infer_smiles[max_pKa_mol] and \
                                small_mol['Index'] == infer_centers[max_pKa_mol]:
                            break
                        small_mol['Index'] = infer_centers[max_pKa_mol]
                    else:
                        if 'Index' not in small_mol.keys():
                            small_mol['Index'] = []
                        small_mol['Index'].append(infer_centers[max_pKa_mol])

                    small_mol['Smiles'] = infer_smiles[max_pKa_mol]
                    ionization_state = infer_ionization_states[0][max_pKa_mol]
                    all_infer_predicts.append(infer_predicts[max_pKa_mol])
                    all_infer_labels.append(infer_labels[max_pKa_mol])
                    all_infer_smiles.append(infer_smiles[max_pKa_mol])
                    all_infer_smiles_base.append(infer_smiles_base[max_pKa_mol])
                    all_neutral.append(infer_neutral[max_pKa_mol])
                    all_infer_mol_num.append(infer_mol_num[max_pKa_mol])
                    all_infer_centers.append(infer_centers[max_pKa_mol])
                    all_infer_proposed_centers.append(infer_proposed_centers[max_pKa_mol])
                    all_infer_ionization_states.append(infer_ionization_states[0][max_pKa_mol])

                    if args.verbose > 1:
                        print("|        | round #%2s: %-102s |" % (protonation_step, small_mol['Smiles']))

                    initial = False
                    infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
                        infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles = \
                            infer(i, small_mol, initial, ionized_smiles, ionization_state, infer_path, model_params,
                                  device, best_hypers, loss_fn, args)

                if len(infer_predicts) == 0:
                    if args.verbose > 1:
                        print('|        | no acid/base pair found                                                                                           |')
                    break
                elif args.verbose > 1:
                    print_inference(infer_predicts, infer_labels, infer_smiles, ionized_smiles, infer_mol_num, ionized_mol_num, infer_centers,
                                    initial_proposed_center, initial, args)

            if len(infer_predicts) > 0:
                for item in range(len(infer_predicts)):
                    if len(all_infer_smiles) == 0 or all_infer_smiles[len(all_infer_smiles) - 1] != infer_smiles[item] \
                            or all_infer_centers[len(all_infer_smiles)-1] != infer_centers[item]:
                        all_infer_predicts.append(infer_predicts[item])
                        all_infer_labels.append(infer_labels[item])
                        all_infer_smiles.append(infer_smiles[item])
                        all_infer_mol_num.append(infer_mol_num[item])
                        all_infer_centers.append(infer_centers[item])
                        all_infer_proposed_centers.append(infer_proposed_centers[item])
                        all_infer_ionization_states.append(infer_ionization_states[0][item])
                        all_infer_ionization_states.append(infer_ionization_states[0][item])

        if args.verbose > 1 and len(all_infer_smiles) > 0:
            print("|        | Final: %-91s----------------|" % (ionized_smiles))
            #Change all_infer_smiles[0] to ionized_smiles, because it reported the first protonated stated after final and not the fully deprotonated one

        print_inference(all_infer_predicts, all_infer_labels, all_infer_smiles, ionized_smiles, all_infer_mol_num, ionized_mol_num,
                        all_infer_centers, initial_proposed_center, initial, args)

        for item in range(len(all_infer_predicts)):
            library_infer_predicts.append(all_infer_predicts[item])
            library_infer_labels.append(all_infer_labels[item])
            library_infer_smiles.append(all_infer_smiles[item])
            library_infer_mol_num.append(all_infer_mol_num[item])
            library_infer_centers.append(all_infer_centers[item])
            library_infer_proposed_centers.append(all_infer_proposed_centers[item])
            library_infer_ionization_states.append(all_infer_ionization_states[item])

    if args.mode == 'test':
        calculate_metrics(library_infer_predicts, library_infer_labels, library_infer_mol_num, args)
        print_results_test(library_infer_predicts, library_infer_labels, library_infer_smiles,
                           library_infer_centers, library_infer_proposed_centers, library_infer_mol_num, args)


def infer(i, small_mol, initial, ionized_smiles, ionization_states, infer_path, model_params, device, best_hypers, loss_fn, args):

    infer_dataset, ionized_smiles = generate_infersets(small_mol, i, initial, ionized_smiles, ionization_states, args)
    dump_datasets(infer_dataset, infer_path)

    # If no acid/base pair found, we exit
    if len(infer_dataset) == 0:
        infer_predicts = []
        infer_labels = []
        infer_smiles = []
        infer_smiles_base = []
        infer_mol_num = []
        infer_centers = []
        infer_proposed_centers = []
        infer_neutrals = []
        infer_ionization_states = []
        return infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, \
            infer_centers, infer_proposed_centers, infer_neutrals, infer_ionization_states, ionized_smiles

    # Loading data for training
    infer_data = load_data(args.infer_pickled)

    if args.GATv2Conv_Or_Other == "GATv2Conv":
        model_infer = GNN(feature_size=infer_dataset[0].x.shape[1],
                          edge_dim=infer_dataset[0].edge_attr.shape[1],
                          model_params=model_params)
    else:
        model_infer = GNN_New(feature_size=infer_dataset[0].x.shape[1],
                          edge_dim=infer_dataset[0].edge_attr.shape[1],
                          model_params=model_params)

    checkpoint = torch.load(args.model_dir + args.model_name, map_location=torch.device('cpu'), weights_only= True)
    model_infer.load_state_dict(checkpoint['model_state_dict'])
    model_infer.eval()

    infer_loader = DataLoader(infer_data, best_hypers["batch_size"],
                              num_workers=0, shuffle=False)

    infer_loss, infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_centers, infer_proposed_centers,\
        infer_mol_num, infer_neutral, infer_error, infer_ionization_states = \
        final_test(model=model_infer, loader=infer_loader, loss_fn=loss_fn, args=args)

    return infer_predicts, infer_labels, infer_smiles, infer_smiles_base, infer_mol_num, infer_centers, \
        infer_proposed_centers, infer_neutral, infer_ionization_states, ionized_smiles


def testing_with_IC(args):
    test_file = args.data_path + args.input
    test_dataset = generate_datasets(test_file, 'Test', args)
    test_path = args.test_pickled
    dump_datasets(test_dataset, test_path)

    device = torch.device("cpu")

    best_hypers = {
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'scheduler_gamma': args.scheduler_gamma,
        'model_embedding_size': args.embedding_size,
        'model_gnn_layers': args.n_graph_layers,
        'model_fc_layers': args.n_FC_layers,
        'model_dropout_rate': 0,
        'model_dense_neurons': args.model_dense_neurons,
        'model_attention_heads': args.model_attention_heads,
    }

    model_params = {k: v for k, v in best_hypers.items() if k.startswith("model_")}
    loss_fn = torch.nn.MSELoss()

    if args.GATv2Conv_Or_Other == "GATv2Conv":
        model_test = GNN(feature_size=test_dataset[0].x.shape[1],
                         edge_dim=test_dataset[0].edge_attr.shape[1],
                         model_params=model_params)
    else:
        model_test = GNN_New(feature_size=test_dataset[0].x.shape[1],
                         edge_dim=test_dataset[0].edge_attr.shape[1],
                         model_params=model_params)

    print('| Reading the files, preparing the features and computing pKa                                                                |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')

    if args.test_data != "none":
        test_data = load_data(args.test_pickled)
    else:
        test_data = load_data(args.train_pickled)
    test_loader = DataLoader(test_data, best_hypers["batch_size"],
                              num_workers=0, shuffle=False)

    checkpoint = torch.load(args.model_dir + args.model_name, map_location=torch.device('cpu'))
    model_test.load_state_dict(checkpoint['model_state_dict'])
    model_test.eval()

    print('| mol #  | SMILES                                                                                    |          pKa          | ')
    print('|        |                                                                                           |  obs. | pred. | error |')
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    test_loss, test_predicts, test_labels, test_smiles, test_smiles_base, test_centers, test_proposed_centers, \
        test_mol_num, test_neutral, test_error, test_ionization_states = \
        final_test(model=model_test, loader=test_loader, loss_fn=loss_fn, args=args)

    test_predicts, test_labels, test_smiles, test_mol_num, test_error = \
        average(test_predicts, test_labels, test_smiles, test_mol_num, test_error, args)

    for i in range(len(test_predicts)):
        print('| %6.0f | %-89s | %5.2f | %5.2f | %5.2f |' % (test_mol_num[i], test_smiles[i], test_labels[i], test_predicts[i],
                                                             abs(test_labels[i]-test_predicts[i])))
    print('|----------------------------------------------------------------------------------------------------------------------------|')
    print('| test loss %-106.3f       |' % test_loss)
    number_of_graphs = args.n_random_smiles
    if number_of_graphs == 0:
        number_of_graphs = 1

    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
    print('| Testing Set averaged over original and %3s random smiles  %-60s     |' % (str(int(number_of_graphs-1)), ' '))
    calculate_metrics(test_predicts, test_labels, test_mol_num, args)
    print('|----------------------------------------------------------------------------------------------------------------------------|', flush=True)
