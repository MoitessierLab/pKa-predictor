
import torch
import numpy as np


def train(epoch, model, train_loader, optimizer, loss_fn):
    # Enumerate over the data
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    running_loss = 0.0
    step = 0
    all_preds = []
    all_labels = []

    for batch in train_loader:

        # Use GPU
        batch.to(device)  

        # Reset gradients
        optimizer.zero_grad()

        # Passing the node features and the connection info
        pred = model(batch.x.float(),
                     batch.edge_index,
                     batch.edge_attr.float(), 
                     batch.node_index,
                     batch.mol_formal_charge,
                     batch.center_formal_charge,
                     batch.batch)
        
        # Calculating the loss and gradients
        loss = loss_fn(torch.squeeze(pred), batch.y.float())
        loss.backward()
        optimizer.step()

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()

    return running_loss/step


def evaluate(epoch, model, val_loader, loss_fn):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    running_loss = 0.0
    step = 0
    all_preds = []
    all_labels = []
    for batch in val_loader:
        batch.to(device)

        pred = model(batch.x.float(), 
                     batch.edge_index,
                     batch.edge_attr.float(), 
                     batch.node_index,
                     batch.mol_formal_charge,
                     batch.center_formal_charge,
                     batch.batch)

        loss = loss_fn(torch.squeeze(pred), batch.y.float())

        # Update tracking
        running_loss += loss.item()
        step += 1
        all_preds.append(torch.squeeze(pred).cpu().detach().numpy())
        all_labels.append(batch.y.cpu().detach().numpy())

    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()   

    return running_loss/step
