import numpy as np
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from dgl.nn.pytorch import GraphConv
from dgl.data.utils import save_graphs, load_graphs
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocsr()

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(features)


adj = np.load('data/ind.20ng.adj', allow_pickle=True)
adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
adj = normalize_adj(adj+sp.eye(adj.shape[0]))
print(adj.sum())
tx = np.load('data/ind.20ng.tx', allow_pickle=True)
ty = np.load('data/ind.20ng.ty', allow_pickle=True)
allx = np.load('data/ind.20ng.allx', allow_pickle=True).toarray()
ally = np.load('data/ind.20ng.ally', allow_pickle=True).toarray()
n_training_docs = int(ally.sum())
n_training_samples = allx.shape[0]

assert(n_training_docs == ally[:n_training_docs+1].sum())
assert(adj.shape[0] - n_training_samples == tx.shape[0])
assert(ally[n_training_docs:-tx.shape[0]].sum() == 0)

train_features = normalize_features(allx)
test_features = normalize_features(np.concatenate([allx[n_training_docs:], tx], 0))
if os.path.isfile('graph.bin'):
    Gs, labels = load_graphs('graph.bin')
else:
    train_G = nx.from_scipy_sparse_matrix(adj[:n_training_samples][:, :n_training_samples])
    train_DGL = dgl.DGLGraph()
    train_DGL.from_networkx(train_G, edge_attrs=['weight'])
    # train_DGL.from_scipy_sparse_matrix(adj[:n_training_samples][:, :n_training_samples])
    assert(len(train_DGL) == train_features.shape[0])

    test_G = nx.from_scipy_sparse_matrix(adj[n_training_docs:][:, n_training_docs:])
    test_DGL = dgl.DGLGraph()
    test_DGL.from_networkx(test_G, edge_attrs=['weight'])
    # test_DGL.from_scipy_sparse_matrix(adj[n_training_docs:][:, n_training_docs:])
    assert(len(test_DGL) == test_features.shape[0])

    Gs = [train_DGL, test_DGL]
    save_graphs('graph.bin', Gs)
print(Gs[0])
print('load graph done')


class Model(nn.Module):
    def __init__(self, feature_dim, inter_dim, final_dim):
        super(Model, self).__init__()
        self.gcn1 = GraphConv(feature_dim, inter_dim)
        self.gcn2 = GraphConv(inter_dim, final_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, graph, features):
        x = self.gcn1(graph, features)
        x = self.dropout(x)
        x = F.relu(x)
        x = self.gcn2(graph, x)
        return F.log_softmax(x, dim=1)


device = 'cuda:0'
model = Model(allx.shape[1], 200, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.02)
train_features = torch.from_numpy(train_features).float().to(device)
test_features = torch.from_numpy(test_features).float().to(device)
ally = torch.from_numpy(ally).argmax(1).to(device)
ty = torch.from_numpy(ty).argmax(1).to(device)
# print(Gs[0].in_degrees().sum(), Gs[0].out_degrees().sum())
# exit()

print(Gs[0].edata['weight'].sum())
Gs[0] = Gs[0].to(torch.device(device))
print(Gs[1].edata['weight'].sum())
Gs[1] = Gs[1].to(torch.device(device))

for epoch in range(500):
    model.train()
    optimizer.zero_grad()
    output = model(Gs[0], train_features)
    loss = F.nll_loss(output[:n_training_docs], ally[:n_training_docs])
    loss.backward()
    optimizer.step()
    print(loss.item())
    if epoch % 10 == 0:
        with torch.no_grad():
            model.eval()
            output = model(Gs[1], test_features)
            pred_y = output.argmax(1)[-ty.shape[0]:]
            print((pred_y == ty).long().sum().float()/float(ty.size(0)))
