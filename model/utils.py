import numpy as np
import pickle as pkl
import networkx as nx
from networkx.readwrite import json_graph
import scipy.sparse as sp
import pandas as pd
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

def load_node2vec_embedding(file):
    with open(file) as f:
        data = f.readlines()
    embed = np.zeros(shape=tuple(map(int,data[0].split(" "))))
    for line in data[1:]:
        temp = list(map(float, line.split(" ")))
        embed[int(temp[0]),:] = temp[1:]
    return embed


def reshape(features):
    return np.hstack(features).reshape((len(features),len(features[0])))

def compute_single_roc(preds, labels):
    score = []
    for i in range(preds.shape[1]):
        fpr, tpr, _ = roc_curve(labels[:,i], preds[:,i])
        roc_auc = auc(fpr, tpr)
        score.append(roc_auc)
    return score

def compute_top_k(preds, labels, k=10):
    total_labels = np.sum(labels)
    total_predicted = 0
    pred_index = np.argsort(-preds)[:,:k]
    for i in range(preds.shape[0]):
        predicted = set(pred_index[i]) & set(np.where(labels[i] == 1)[0])
        total_predicted += len(predicted)
    
    return total_labels, total_predicted

def calculate_accuracy(y_test, y_score):
    y_score_max = np.argmax(y_score,axis=1)
    cnt = 0
    for i in range(y_score.shape[0]):
        if y_test[i, y_score_max[i]] == 1:
            cnt += 1
    
    total_genes_with_labels = sum(y_test.sum(axis=1) != 0)
    
    return float(cnt)/total_genes_with_labels

def evaluate_performance(y_test, y_score):
    """Evaluate performance"""
    n_classes = y_test.shape[1]
    perf = dict()
    
    perf["M-aupr"] = 0.0
    n = 0
    for i in range(n_classes):
        ap = average_precision_score(y_test[:, i], y_score[:, i])
        if sum(y_test[:, i]) > 0:
            n += 1
            perf["M-aupr"] += ap
    perf["M-aupr"] /= n

    # Compute micro-averaged AUPR
    perf['m-aupr'] = average_precision_score(y_test.ravel(), y_score.ravel())

    # Computes accuracy
    perf['accuracy'] = calculate_accuracy(y_test, y_score)

    # Computes F1-score
    alpha = 3
    y_new_pred = np.zeros_like(y_test)
    for i in range(y_test.shape[0]):
        top_alpha = np.argsort(y_score[i, :])[-alpha:]
        y_new_pred[i, top_alpha] = np.array(alpha*[1])
    perf["F1-score"] = f1_score(y_test, y_new_pred, average='micro')

    return perf
    

def compute_performance(preds, labels):
    preds = np.round(preds, 2)
    labels = labels.astype(np.int32)
    f_max = 0
    p_max = 0
    r_max = 0
    sp_max = 0
    t_max = 0
    for t in range(1, 100):
        threshold = t / 100.0
        predictions = (preds > threshold).astype(np.int32)
        tp = np.sum(predictions * labels)
        fp = np.sum(predictions) - tp
        fn = np.sum(labels) - tp
        sn = tp / (1.0 * np.sum(labels))
        sp = np.sum((predictions ^ 1) * (labels ^ 1))
        sp /= 1.0 * np.sum(labels ^ 1)
        fpr = 1 - sp
        precision = tp / (1.0 * (tp + fp))
        recall = tp / (1.0 * (tp + fn))
        f = 2 * precision * recall / (precision + recall)
        if f_max < f:
            f_max = f
            p_max = precision
            r_max = recall
            sp_max = sp
            t_max = threshold
    return f_max, p_max, r_max, sp_max, t_max