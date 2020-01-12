import pandas as pd
import numpy as np
import networkx as nx

import networkx as nx
from node2vec import Node2Vec

feats = pd.read_pickle("../data/final_input/features.pkl")

print("running node2vec...")

#G_new = nx.from_scipy_sparse_matrix(sAdj_one)
G_new = nx.from_scipy_sparse_matrix(sAdj_embed)

node2vec_new = Node2Vec(G_new, dimensions=256, walk_length=20, num_walks=200, workers=28)
model_new = node2vec_new.fit(window=10, min_count=1, batch_words=4)  
model_new.wv.save_word2vec_format("../data/final_input/node2vec_coexpr/node2vec_coexpr_" + args.cancer + ".txt")


