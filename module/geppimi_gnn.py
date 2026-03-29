import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from ogb.utils.features import atom_to_feature_vector, bond_to_feature_vector
from torch_geometric.data import Data
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree


def smiles_to_pyg(smiles, mol_from_smiles):
    """Convert a SMILES string into a PyG Data object with atom/bond indices."""
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None

    atom_features_list = [atom_to_feature_vector(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    if len(mol.GetBonds()) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 3), dtype=torch.long)
    else:
        edges = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_feature = bond_to_feature_vector(bond)
            edges.append((i, j))
            edges.append((j, i))
            edge_attrs.append(edge_feature)
            edge_attrs.append(edge_feature)
        edge_index = torch.tensor(np.array(edges).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_attrs), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data


class GINConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim),
        )
        self.eps = nn.Parameter(torch.Tensor([0]))
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)


class GCNConv(MessagePassing):
    def __init__(self, emb_dim, aggr="add"):
        super().__init__(aggr=aggr)
        self.linear = nn.Linear(emb_dim, emb_dim)
        self.root_emb = nn.Embedding(1, emb_dim)
        self.bond_encoder = BondEncoder(emb_dim=emb_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        edge_embedding = self.bond_encoder(edge_attr)
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return self.propagate(edge_index, x=x, edge_attr=edge_embedding, norm=norm) + F.relu(
            x + self.root_emb.weight
        ) * 1.0 / deg.view(-1, 1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)


class GNNEncoder(nn.Module):
    """Graph encoder adapted from MultiPPIMI (GIN/GCN backbone)."""

    def __init__(self, num_layer=5, emb_dim=300, JK="last", drop_ratio=0.0, gnn_type="gin"):
        super().__init__()
        if num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK

        self.atom_encoder = AtomEncoder(emb_dim)
        self.gnns = nn.ModuleList()
        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim, aggr="add"))
            elif gnn_type == "gcn":
                self.gnns.append(GCNConv(emb_dim, aggr="add"))
            else:
                raise ValueError(f"Unsupported gnn_type {gnn_type}")

        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(emb_dim) for _ in range(num_layer)])

    @property
    def out_dim(self):
        return self.emb_dim if self.JK != "concat" else (self.num_layer + 1) * self.emb_dim

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.atom_encoder(x)

        h_list = [x]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)
            h_list.append(h)

        if self.JK == "concat":
            node_representation = torch.cat(h_list, dim=1)
        elif self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "max":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.max(torch.cat(h_list, dim=0), dim=0)[0]
        elif self.JK == "sum":
            h_list = [h.unsqueeze_(0) for h in h_list]
            node_representation = torch.sum(torch.cat(h_list, dim=0), dim=0)[0]
        else:
            raise ValueError("JK must be one of ['concat', 'last', 'max', 'sum'].")
        return node_representation

    def load_from_file(self, model_file, map_location="cpu"):
        state = torch.load(model_file, map_location=map_location)
        self.load_state_dict(state)
        return self
