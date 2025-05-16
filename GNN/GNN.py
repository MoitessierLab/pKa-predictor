
import torch
import torch.nn.functional as F

from torch.nn import Linear, BatchNorm1d, LayerNorm, ModuleList
from torch_geometric.nn import AttentionalAggregation, GATv2Conv, TransformerConv, GlobalAttention


class GNN(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params):
        super(GNN, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        self.gnn_layers = model_params["model_gnn_layers"]
        self.dense_layers = model_params["model_fc_layers"]
        self.p = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        n_heads = model_params["model_attention_heads"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.fc_layers = ModuleList([])

        # GNN Layers
        self.conv1 = GATv2Conv(feature_size, 
                               embedding_size,
                               heads=n_heads,
                               edge_dim=edge_dim,
                               dropout=self.p,
                               concat=True)
        
        self.transf1 = Linear(embedding_size*n_heads, embedding_size)
        
        self.bn1 = BatchNorm1d(embedding_size)

        for i in range(self.gnn_layers-1):
            self.conv_layers.append(GATv2Conv(embedding_size,
                                              embedding_size,
                                              heads=n_heads,
                                              edge_dim=edge_dim,
                                              dropout=self.p,
                                              concat=True))

            self.transf_layers.append(Linear(embedding_size*n_heads, embedding_size))
            
            self.bn_layers.append(BatchNorm1d(embedding_size))
            
        self.att = AttentionalAggregation(Linear(embedding_size, 1))
        
        # Linear layers the formal charges (molecule and ionization center) will be added at this stage
        # And acid and base embeddings will be concatenated
        self.linear1 = Linear(embedding_size + 2, dense_neurons)

        for i in range(self.dense_layers-1):
            self.fc_layers.append(Linear(dense_neurons, int(dense_neurons/4)))
            dense_neurons = int(dense_neurons/4)
        
        self.out_layer = Linear(dense_neurons, 1) 

    def forward(self, x, edge_index, edge_attr, node_index, mol_formal_charge, center_formal_charge, batch_index):
        # At this stage, x is a single tensor containing all the atoms of all the batch molecules. The references to
        # the corresponding molecules are given in batch_index
        #
        # Initial GATv2Conv transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = self.bn1(x)

        # now layers of GATv2Conv
        for i in range(self.gnn_layers-1):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        # Removing all the atoms' molecule indexes (which atom is in which molecule) in the batch not part of the local environment.
        # But are still all there. We remove then from the node features
        # x may contain several molecules (the node indexes are for all of them)
        x = x[node_index]
        # And update the batch_index (tensor like: 0 0 0 0 0 1 1 1 1... indicating first 5 atoms are from the first
        # molecule,...
        mask = torch.zeros(batch_index.numel(), dtype=torch.bool)
        mask[node_index] = True
        batch_index = batch_index[mask]
        # Attention
        x = self.att(x, batch_index)

        # We append formal charge to the embedding
        # formal_charge shape is [batch_size*2], we want it [batch_size, 2]
        # We first add a dimension then transpose
        mol_formal_charge = mol_formal_charge[:, None]
        torch.transpose(mol_formal_charge, 0, 1)
        x = torch.cat([x, mol_formal_charge], axis=1)

        center_formal_charge = center_formal_charge[:, None]
        torch.transpose(center_formal_charge, 0, 1)

        # If we subtract, the element, electronegativity,... will be lost
        x = torch.cat([x, center_formal_charge], axis=1)

        x = torch.relu(self.linear1(x))

        x = F.dropout(x, p=self.p)

        for i in range(self.dense_layers-1):
            x = torch.relu(self.fc_layers[i](x))
            x = F.dropout(x, p=self.p)

        x = self.out_layer(x)

        return x


class GNN_New(torch.nn.Module):
    def __init__(self, feature_size, edge_dim, model_params):
        super(GNN_New, self).__init__()
        embedding_size = model_params["model_embedding_size"]
        self.gnn_layers = model_params["model_gnn_layers"]
        self.dense_layers = model_params["model_fc_layers"]
        self.p = model_params["model_dropout_rate"]
        dense_neurons = model_params["model_dense_neurons"]
        n_heads = model_params["model_attention_heads"]

        self.conv_layers = ModuleList([])
        self.transf_layers = ModuleList([])
        self.bn_layers = ModuleList([])
        self.fc_layers = ModuleList([])

        # GNN Layers
        self.conv1 = TransformerConv(in_channels=feature_size,
                                     out_channels=embedding_size,
                                     heads=n_heads,
                                     dropout=self.p,
                                     edge_dim=edge_dim,
                                     concat=True)

        self.transf1 = Linear(embedding_size * n_heads, embedding_size)

        self.bn1 = LayerNorm(embedding_size)

        for i in range(self.gnn_layers):
            self.conv_layers.append(TransformerConv(in_channels=embedding_size,
                                                    out_channels=embedding_size,
                                                    heads=n_heads,
                                                    dropout=self.p,
                                                    edge_dim=edge_dim,
                                                    concat=True))
            self.transf_layers.append(Linear(embedding_size * n_heads, embedding_size))
            self.bn_layers.append(LayerNorm(embedding_size))

        self.att = GlobalAttention(gate_nn=Linear(embedding_size, 1))

        # Linear layers the formal charges (molecule and ionization center) will be added at this stage
        # And acid and base embeddings will be concatenated
        self.linear1 = Linear(embedding_size + 2, dense_neurons)

        for i in range(self.dense_layers-1):
            self.fc_layers.append(Linear(dense_neurons, int(dense_neurons / 4)))
            dense_neurons = int(dense_neurons / 4)

        self.out_layer = Linear(dense_neurons, 1)

    def forward(self, x, edge_index, edge_attr, node_index, mol_formal_charge, center_formal_charge, batch_index):
        # At this stage, x is a single tensor containing all the atoms of all the batch molecules. The references to
        # the corresponding molecules are given in batch_index

        # Initial GAT transformation
        x = self.conv1(x, edge_index, edge_attr)
        x = torch.relu(self.transf1(x))
        x = F.dropout(x, p=self.p, training=self.training)
        x = self.bn1(x)

        # now layers of GAT
        for i in range(self.gnn_layers - 1):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = torch.relu(self.transf_layers[i](x))
            x = self.bn_layers[i](x)

        x = x[node_index]

        mask = torch.zeros(batch_index.numel(), dtype=torch.bool)
        mask[node_index] = True
        batch_index = batch_index[mask]
        # Attention
        x = self.att(x, batch_index)

        mol_formal_charge = mol_formal_charge[:, None]
        torch.transpose(mol_formal_charge, 0, 1)
        x = torch.cat([x, mol_formal_charge], axis=1)

        # If we subtract, the element, electronegativity,... will be lost
        center_formal_charge = center_formal_charge[:, None]
        torch.transpose(center_formal_charge, 0, 1)
        x = torch.cat([x, center_formal_charge], axis=1)

        x = torch.relu(self.linear1(x))
        x = F.dropout(x, p=self.p)

        for i in range(self.dense_layers - 1):
            x = torch.relu(self.fc_layers[i](x))
            x = F.dropout(x, p=self.p)

        x = self.out_layer(x)

        return x
