import torch
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
from temp_gntk import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import torch_scatter
from utils_graph_classification import *
from grakel.kernels import WeisfeilerLehman, ShortestPathAttr, ShortestPath, RandomWalkLabeled
from grakel import Graph
from karateclub.graph_embedding import Graph2Vec, NetLSD, GL2Vec

import os
import time

import numpy as np
import torch
import torch_geometric
import pandas as pd
import math
import matplotlib.pyplot as plt
from tgb.linkproppred.dataset_pyg import PyGLinkPropPredDataset
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from torch_geometric.loader.temporal_dataloader import TemporalDataLoader, TemporalData
# from utils import *
# from convergence import *
import pdb

class args():
    time_dim = 25
    alpha = np.sqrt(time_dim)
    beta = np.sqrt(time_dim)
    num_sub_graphs = 5
    k_recent = 15
    num_mlp_layers = 1
    device = "cpu"
    node_ntk = False
    encode_time = True 
    relative_difference = True
    neighborhood_avg = False
    node_onehot = False
    mean_graph_pooling = False
    jumping_knowledge = False
    skip_connection = False

    if encode_time is False:
        time_dim = 1

data_dirs = ["./datasets/infectious_ct1/infectious_ct1", 
             "./datasets/dblp_ct1/dblp_ct1",
             "./datasets/facebook_ct1/facebook_ct1",
             "./datasets/tumblr_ct1/tumblr_ct1",
             "./datasets/highschool_ct1/highschool_ct1"]

data_names = ["infectious_ct1", "dblp_ct1", "facebook_ct1", "tumblr_ct1", "highschool_ct1"]

def t_gntk(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    train_size = 100
    # int(0.7 * len(temporal_graphs))
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)
    gram_matrix = compute_gram_matrix((graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args)

    for subgraph_id in range(args.num_sub_graphs):
        train_gram = np.copy(gram_matrix[:, :, subgraph_id])
        train_gram /= train_gram[train_gram != 0].min()
        cv_test, cv_train, clf = search(np.nan_to_num(train_gram), labels, C_list)
        
        gram_norm = np.copy(train_gram)
        diag = np.sqrt(np.diag(gram_norm))
        gram_norm /= (diag[:, None] * diag[None, :])
        cv_test_norm, cv_train_norm, clf_norm = search(np.nan_to_num(gram_norm), labels, C_list)

        for score, clf in zip([cv_test, cv_test_norm], [clf, clf_norm]):
            if best_score < score:
                best_score = score
                best_clf = clf
                        
        # print(f'CV score of {"t_gntk"}, {subgraph_id}: test {cv_test:.4f}, train {cv_train:.4f}')
        # print(f'CV score of {"t_gntk"} (norm), {subgraph_id}: test {cv_test:.4f}, train {cv_train:4f}')
        # print("\n")
    
    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"t_gntk"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"t_gntk"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def wl(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    train_size = 100
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    wl_norm = WeisfeilerLehman(normalize = True)
    wl = WeisfeilerLehman()
    ###

    time_id = 10
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)
    for subgraph_id in range(args.num_sub_graphs):
        # best_subgraph_wl, best_subgraph_sp, best_subgraph_rd = 0.0, 0.0, 0.0
        # std_subgraph_wl, std_subgraph_sp, std_subgraph_rd = 0.0, 0.0, 0.0 
        
        grakel_adjs, grakel_attrs = grakel_graphs((graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args, num_slice = subgraph_id + 1)
        k_train = []
        for graph_id in range(len(grakel_adjs)):
            G = Graph(initialization_object = grakel_adjs[graph_id], node_labels = grakel_attrs[time_id][graph_id])
            k_train.append(G)

        for grakel_kernel, kernel_name in zip([wl, wl_norm], ["WL", "WL norm"]):
            # pdb.set_trace()
            gram_matrix = grakel_kernel.fit_transform(k_train)
            cv_test, cv_train, grakel_clf = search(np.nan_to_num(gram_matrix), labels, C_list)
            # if cv_test > best_subgraph_wl:
            #     best_subgraph_wl = cv_test
            #     std_subgraph_wl = grakel_clf.cv_results_["std_test_score"][grakel_clf.cv_results_["mean_test_score"] == grakel_clf.cv_results_["mean_test_score"].max()].min()
            if cv_test > best_score:
                best_score = cv_test
                best_clf = grakel_clf
            # print("CV score of {}, {}: test {}, train {}".format(kernel_name, subgraph_id, cv_test, cv_train))
        
        # print("Best CV Score of {}, {}: {}; {}".format("WL", subgraph_id, best_subgraph_wl, std_subgraph_wl))

    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"WL"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"WL"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def sp(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    train_size = 100
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    sp_norm = ShortestPath(with_labels = True, normalize = True)
    sp = ShortestPath()
    ###

    time_id = 10
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)
    
    for subgraph_id in range(args.num_sub_graphs):
        # best_subgraph_wl, best_subgraph_sp, best_subgraph_rd = 0.0, 0.0, 0.0
        # std_subgraph_wl, std_subgraph_sp, std_subgraph_rd = 0.0, 0.0, 0.0 
        
        grakel_adjs, grakel_attrs = grakel_graphs((graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args, num_slice = subgraph_id + 1)
        k_train = []
        for graph_id in range(len(grakel_adjs)):
            G = Graph(initialization_object = grakel_adjs[graph_id], node_labels = grakel_attrs[time_id][graph_id])
            k_train.append(G)

        for grakel_kernel, kernel_name in zip([sp, sp_norm], ["SP", "SP norm"]):
            # pdb.set_trace()
            gram_matrix = grakel_kernel.fit_transform(k_train)
            cv_test, cv_train, grakel_clf = search(np.nan_to_num(gram_matrix), labels, C_list)
            # if cv_test > best_subgraph_wl:
            #     best_subgraph_wl = cv_test
            #     std_subgraph_wl = grakel_clf.cv_results_["std_test_score"][grakel_clf.cv_results_["mean_test_score"] == grakel_clf.cv_results_["mean_test_score"].max()].min()
            if cv_test > best_score:
                best_score = cv_test
                best_clf = grakel_clf
            # print("CV score of {}, {}: test {}, train {}".format(kernel_name, subgraph_id, cv_test, cv_train))
        
        # print("Best CV Score of {}, {}: {}; {}".format("WL", subgraph_id, best_subgraph_wl, std_subgraph_wl))

    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"SP"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"SP"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def rd(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)
    train_size = 100
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    rd_norm = RandomWalkLabeled(normalize = True)
    rd = RandomWalkLabeled()
    ###

    time_id = 10
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)
    
    for subgraph_id in range(args.num_sub_graphs):
        # best_subgraph_wl, best_subgraph_sp, best_subgraph_rd = 0.0, 0.0, 0.0
        # std_subgraph_wl, std_subgraph_sp, std_subgraph_rd = 0.0, 0.0, 0.0 
        
        grakel_adjs, grakel_attrs = grakel_graphs((graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args, num_slice = subgraph_id + 1)
        k_train = []
        for graph_id in range(len(grakel_adjs)):
            G = Graph(initialization_object = grakel_adjs[graph_id], node_labels = grakel_attrs[time_id][graph_id])
            k_train.append(G)

        for grakel_kernel, kernel_name in zip([rd, rd_norm], ["RD", "RD norm"]):
            # pdb.set_trace()
            gram_matrix = grakel_kernel.fit_transform(k_train)
            cv_test, cv_train, grakel_clf = search(np.nan_to_num(gram_matrix), labels, C_list)
            # if cv_test > best_subgraph_wl:
            #     best_subgraph_wl = cv_test
            #     std_subgraph_wl = grakel_clf.cv_results_["std_test_score"][grakel_clf.cv_results_["mean_test_score"] == grakel_clf.cv_results_["mean_test_score"].max()].min()
            if cv_test > best_score:
                best_score = cv_test
                best_clf = grakel_clf
            # print("CV score of {}, {}: test {}, train {}".format(kernel_name, subgraph_id, cv_test, cv_train))
        
        # print("Best CV Score of {}, {}: {}; {}".format("WL", subgraph_id, best_subgraph_wl, std_subgraph_wl))

    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"RD"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"RD"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def graph2vec(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    train_size = 100
    # int(0.7 * len(temporal_graphs))
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    embedder = Graph2Vec()
    ###
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)

    nx_graph_lists = []
    # test_nx_graph_lists = []
    for subgraph_id in range(1, 6):
        nx_graph_list = []
        for graph_adjs in graphs_adjs:
            adj = graph_adjs[-subgraph_id]
            nx_graph_list.append(adj_to_nxgraph(adj))
        nx_graph_lists.append(nx_graph_list)
    # gram_matrix = compute_gram_matrix((graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args)
    start_time = time.time()
    for subgraph_id in range(args.num_sub_graphs):
        nx_graph_list = nx_graph_lists[subgraph_id - 1]
            # test_nx_graph_list = test_nx_graph_lists[subgraph_id - 1]

        embedder.fit(nx_graph_list)
        graph_embs = embedder.get_embedding()
        # test_embs = embedder.infer(test_nx_graph_list)

        gram_matrix = compute_gram_matrix_from_graph_embs(graph_embs)
        gram_matrix = np.nan_to_num(gram_matrix)
        norm = compute_gram_matrix_from_graph_embs(graph_embs, normalize = True)
        norm = np.nan_to_num(norm)

        cv_test, cv_train, clf = search(gram_matrix, labels, C_list)
        cv_test_norm, cv_train_norm, clf_norm = search(norm, labels, C_list)        

        for score, clf in zip([cv_test, cv_test_norm], [clf, clf_norm]):
            if best_score < score:
                best_score = score
                best_clf = clf
                        
    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"Graph2Vec"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"Graph2Vec"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def netlsd(data_dir, data_name):
    print(data_dir)

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    train_size = 100
    # int(0.7 * len(temporal_graphs))
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    embedder = NetLSD()
    ###
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)

    nx_graph_lists = []
    # test_nx_graph_lists = []
    for subgraph_id in range(1, 6):
        nx_graph_list = []
        for graph_adjs in graphs_adjs:
            adj = graph_adjs[-subgraph_id]
            nx_graph_list.append(adj_to_nxgraph(adj))
        nx_graph_lists.append(nx_graph_list)
    # gram_matrix = compute_gram_matrix((graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args)

    for subgraph_id in range(args.num_sub_graphs):
        nx_graph_list = nx_graph_lists[subgraph_id - 1]
            # test_nx_graph_list = test_nx_graph_lists[subgraph_id - 1]

        embedder.fit(nx_graph_list)
        graph_embs = embedder.get_embedding()
        # test_embs = embedder.infer(test_nx_graph_list)

        gram_matrix = compute_gram_matrix_from_graph_embs(graph_embs)
        gram_matrix = np.nan_to_num(gram_matrix)
        norm = compute_gram_matrix_from_graph_embs(graph_embs, normalize = True)
        norm = np.nan_to_num(norm)

        cv_test, cv_train, clf = search(gram_matrix, labels, C_list)
        cv_test_norm, cv_train_norm, clf_norm = search(norm, labels, C_list)        

        for score, clf in zip([cv_test, cv_test_norm], [clf, clf_norm]):
            if best_score < score:
                best_score = score
                best_clf = clf
                        
    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"NetLSD"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"NetLSD"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

def gl2vec(data_dir, data_name):
    print(data_dir)

    start_time = time.time()

    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    train_size = 100
    # int(0.7 * len(temporal_graphs))
    train_ds = temporal_graphs[:train_size]
    labels = graphs_label[:train_size]

    C_list = np.logspace(-2, 4, 120)

    best_score, best_clf = 0.0, None

    args.k_recent = 50000

    ###
    embedder = GL2Vec()
    ###
    start_time = time.time()
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = pre_t_gntk(train_ds, args)

    nx_graph_lists = []
    # test_nx_graph_lists = []
    for subgraph_id in range(1, 6):
        nx_graph_list = []
        for graph_adjs in graphs_adjs:
            adj = graph_adjs[-subgraph_id]
            nx_graph_list.append(adj_to_nxgraph(adj))
        nx_graph_lists.append(nx_graph_list)
    # gram_matrix = compute_gram_matrix((graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs), args)

    for subgraph_id in range(args.num_sub_graphs):
        nx_graph_list = nx_graph_lists[subgraph_id - 1]
            # test_nx_graph_list = test_nx_graph_lists[subgraph_id - 1]

        embedder.fit(nx_graph_list)
        graph_embs = embedder.get_embedding()
        # test_embs = embedder.infer(test_nx_graph_list)

        gram_matrix = compute_gram_matrix_from_graph_embs(graph_embs)
        gram_matrix = np.nan_to_num(gram_matrix)
        norm = compute_gram_matrix_from_graph_embs(graph_embs, normalize = True)
        norm = np.nan_to_num(norm)

        cv_test, cv_train, clf = search(gram_matrix, labels, C_list)
        cv_test_norm, cv_train_norm, clf_norm = search(norm, labels, C_list)        

        for score, clf in zip([cv_test, cv_test_norm], [clf, clf_norm]):
            if best_score < score:
                best_score = score
                best_clf = clf
                        
    total_time = time.time() - start_time

    score_std = None
    for score, std in zip(best_clf.cv_results_["mean_test_score"], best_clf.cv_results_["std_test_score"]):
        if score == best_score:
            score_std = std
            break

    print(f'Run time of {"GL2Vec"} on {data_name}: {total_time:.2f} seconds')
    print(f'CV score of {"GL2Vec"}, test score: {best_score:.4f}, std: {score_std:.4f}')
    return total_time

if __name__ == "__main__":
    # H2O = Graph([[0, 1, 1], [1, 0, 0], [1, 0, 0]], {0: 'O', 1: 'H', 2: 'H'})
    all_runtime = {"wl": [], 
     "sp": [],
     "rd": [],
     "g2v": [],
     "nlsd": [],
     "gl2v": [],
     "tgntk": []}
    for i in range(len(data_dirs)):
        all_runtime["wl"].append(wl(data_dirs[i], data_names[i]))
        all_runtime["sp"].append(sp(data_dirs[i], data_names[i]))
        all_runtime["rd"].append(rd(data_dirs[i], data_names[i]))
        all_runtime["g2v"].append(graph2vec(data_dirs[i], data_names[i]))
        all_runtime["nlsd"].append(netlsd(data_dirs[i], data_names[i]))
        all_runtime["gl2v"].append(gl2vec(data_dirs[i], data_names[i]))
        all_runtime["tgntk"].append(t_gntk(data_dirs[i], data_names[i]))
        # all_runtime["wl"].append(data_dirs[i], data_names[i])

    print(all_runtime)
