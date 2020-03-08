import numpy as np
import networkx as nx
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch import optim
from gcn import GraphConv
from dgl.data.utils import save_graphs, load_graphs

full_adj = np.load('data/ind.20ng.adj', allow_pickle=True).tocsr()
tx = np.load('data/ind.20ng.tx', allow_pickle=True)
ty = np.load('data/ind.20ng.ty', allow_pickle=True)
allx = np.load('data/ind.20ng.allx', allow_pickle=True).toarray()
ally = np.load('data/ind.20ng.ally', allow_pickle=True).toarray()
n_training_docs = int(ally.sum())
n_training_samples = allx.shape[0]

assert(n_training_docs == ally[:n_training_docs+1].sum())
assert(full_adj.shape[0] - n_training_samples == tx.shape[0])
assert(ally[n_training_docs:-tx.shape[0]].sum() == 0)

train_features = allx
test_features = np.concatenate([allx[n_training_docs:], tx], 0)
if os.path.isfile('graph.bin'):
    Gs, labels = load_graphs('graph.bin')
else:
    train_G = nx.from_scipy_sparse_matrix(full_adj[:n_training_samples][:, :n_training_samples])
    train_DGL = dgl.DGLGraph()
    train_DGL.from_networkx(train_G, edge_attrs=['weight'])
    # train_DGL.from_scipy_sparse_matrix(full_adj[:n_training_samples][:, :n_training_samples])
    assert(len(train_DGL) == train_features.shape[0])

    test_G = nx.from_scipy_sparse_matrix(full_adj[n_training_docs:][:, n_training_docs:])
    test_DGL = dgl.DGLGraph()
    test_DGL.from_networkx(test_G, edge_attrs=['weight'])
    # test_DGL.from_scipy_sparse_matrix(full_adj[n_training_docs:][:, n_training_docs:])
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
        # x = self.dropout(x)
        x = F.relu(x)
        x = self.gcn2(graph, x)
        return F.log_softmax(x, dim=1)

device = 'cuda:0'
model = Model(allx.shape[1], 200, 20).to(device)
optimizer = optim.Adam(model.parameters(), lr=2e-2)
train_features = torch.from_numpy(train_features).float().to(device)
test_features = torch.from_numpy(test_features).float().to(device)
ally = torch.from_numpy(ally).argmax(1).to(device)
ty = torch.from_numpy(ty).argmax(1).to(device)
# print(Gs[0].in_degrees().sum(), Gs[0].out_degrees().sum())
# exit()

Gs[0].edata['weight'].unsqueeze_(1)
Gs[0] = Gs[0].to(torch.device(device))
Gs[1].edata['weight'].unsqueeze_(1)
Gs[1] = Gs[1].to(torch.device(device))

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    output = model(Gs[0], train_features)
    loss = F.nll_loss(output[:n_training_docs], ally[:n_training_docs])
    loss.backward()
    optimizer.step()
    print(loss.item())

with torch.no_grad():
    model.eval()
    output = model(Gs[1], test_features)
    pred_y = output.argmax(1)[-ty.shape[0]:]
    print((pred_y == ty).long().sum().float()/float(ty.size(0)))
