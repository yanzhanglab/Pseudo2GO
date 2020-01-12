import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import pandas as pd

import torch
from torch_geometric.data import Data
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GATConv

from utils import load_node2vec_embedding, reshape, compute_single_roc, compute_top_k, compute_performance, calculate_accuracy, evaluate_performance
import argparse
from collections import defaultdict
import json

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='GCN')
parser.add_argument('--label', type=str, default='mf')
parser.add_argument('--hidden', type=int, default=256)
parser.add_argument('--iter', type=int, default=400)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--attribute', type=str, default="all")
parser.add_argument('--network', type=str, default="simi")
args = parser.parse_args()

path = "../data/final_input/"
feats = pd.read_pickle(path + "features.pkl")


class my_GCN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(my_GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, args.hidden, cached=False)
        self.conv2 = GCNConv(args.hidden, out_channels, cached=False)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def get_embedding(self, x, edge_index):
        x = self.conv1(x, edge_index)
        return x
    
def choose_attribute(attr):
    elif attr == "miRNA":
        x = reshape(feats['microRNA_250'].values)
        print("microRNA feature")
    elif attr == "TCGA-coexpr":
        x = load_node2vec_embedding(path + "node2vec_TCGA_BRCA.txt")
        print("TCGA coexpression node2vec feature")
    elif attr == "ppi":
        x = load_node2vec_embedding(path + "node2vec_ppi.txt")
        print("ppi feature")
    elif attr == "GTEx-expression":
        x = reshape(feats['expression'].values)
        print("GTEx median expression")
    elif attr == "GTEx-node2vec":
        x = load_node2vec_embedding(path + "node2vec_GTEx.txt")
        print("GTEx node2vec")
    elif attr == "all":
        x_coexpr = load_node2vec_embedding(path + "node2vec_TCGA_BRCA.txt")
        x_ppi = load_node2vec_embedding(path + "node2vec_ppi.txt")
        x_mirna = reshape(feats['microRNA_250'].values)
        x_GTEx = load_node2vec_embedding(path + "node2vec_GTEx.txt")
        x = np.hstack((x_coexpr, x_ppi, x_mirna, x_GTEx))
        print("microRNA plus coexpresion plus ppi plus GTEx feature")
    return x

def choose_network(netw):
    if netw == "simi":
        edges = sp.load_npz(path + "adj_simi.npz")
        print("similarity network")
    return edges

def choose_label(lab):
    if lab == "cc":
        data_y = torch.tensor(reshape(feats['cc_label']), dtype=torch.float)
        print("CC")
    elif lab == "mf":
        data_y = torch.tensor(reshape(feats['mf_label']), dtype=torch.float)
        print("MF")
    elif lab == "bp":
        data_y = torch.tensor(reshape(feats['bp_label']), dtype=torch.float)
        print("BP")
    return data_y

def load_data():   
    x = choose_attribute(args.attribute)
    edges = choose_network(args.network)
    data_y = choose_label(args.label)
    
    # features
    data_x = torch.tensor(x, dtype=torch.float)
    # edges
    edge_index = edges.tocoo()
    row, col = edge_index.row, edge_index.col
    data_edge_index = torch.tensor([row,col], dtype=torch.long)
    
    data = Data(x=data_x, edge_index=data_edge_index, y=data_y)
    # training on coding genes
    data.train_mask = torch.tensor(feats['gene_type'] == 'protein_coding', dtype=torch.uint8)
    # test on pseudogenes that have at least one GO annotation
    data.test_mask = torch.tensor((feats['gene_type'] != 'protein_coding') & (data.y.cpu().numpy().sum(axis=1) != 0), dtype=torch.uint8)
    print("number of test cases", sum((feats['gene_type'] != 'protein_coding') & (data.y.cpu().numpy().sum(axis=1) != 0)))
    
    return data

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = load_data().to(device)
    model = my_GCN(data.num_features, data.y.size(1)).to(device)
    
    loss_op = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    for epoch in range(1, args.iter+1):
        model.train()
        optimizer.zero_grad()
        loss = loss_op(model(data.x, data.edge_index)[data.train_mask], data.y[data.train_mask])
        print(epoch, loss.item())
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        final_preds = out[data.test_mask].cpu().numpy()
        final_labels = data.y[data.test_mask].cpu().numpy()
        perf = evaluate_performance(final_labels, final_preds)
        
        embeds = model.get_embedding(data.x, data.edge_index).cpu().numpy()
        
    return model, perf, final_preds, final_labels





trained_model, perf, preds, labels = train_model()
filename = "../results/results_" + args.label + "_" + args.attribute + ".json"
with open(filename,"w") as f:
    json.dump(perf, f)
print("--------------------------------")
print("performance", perf)

    
