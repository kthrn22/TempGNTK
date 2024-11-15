import torch
import pandas as pd
import numpy as np

import torch_geometric
import timeit
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt

from torch_geometric.data import TemporalData
from torch_geometric.loader import TemporalDataLoader
from tgb.nodeproppred.dataset_pyg import PyGNodePropPredDataset
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from tgb.nodeproppred.evaluate import Evaluator
from torch_geometric.nn.models.tgn import LastNeighborLoader
import torch_scatter
from temp_gntk import *

# from utils_graph_classification import *

temporal_data = torch.load("./datasets/tgbn_trade/tgbn_trade_data.pt")
dataset = torch.load("./datasets/tgbn_trade/dataset.pt")

train_data = TemporalData(src = temporal_data.src[dataset.train_mask],
                          dst = temporal_data.dst[dataset.train_mask], 
                          t = temporal_data.t[dataset.train_mask],
                          msg = temporal_data.msg[dataset.train_mask],
                          y = temporal_data.y[dataset.train_mask])

val_data = TemporalData(src = temporal_data.src[dataset.val_mask],
                            dst = temporal_data.dst[dataset.val_mask],
                            t = temporal_data.t[dataset.val_mask],
                            msg = temporal_data.msg[dataset.val_mask],
                            y = temporal_data.y[dataset.val_mask])

test_data = TemporalData(src = temporal_data.src[dataset.test_mask],
                         dst = temporal_data.dst[dataset.test_mask],
                         t = temporal_data.t[dataset.test_mask],
                         msg = temporal_data.msg[dataset.test_mask],
                         y = temporal_data.y[dataset.test_mask])

t_gntk = TemporalGNTK()

class args():
    time_dim = 20
    alpha = np.sqrt(time_dim)
    beta = np.sqrt(time_dim)
    num_sub_graphs = 5
    k_recent = 15
    num_mlp_layers = 2
    device = "cuda:0"
    node_ntk = True
    C = 1.0
    time_window = None

# we will do hyperparam search on k_recent, and C
    
def search_best():
    best_val_score = 0.0
    best_clf = None
    best_params = None
    C_list = np.logspace(-1, 2, 60)

    for num_mlp_layers in [3]:
        args.num_mlp_layers = num_mlp_layers

        for k_recent in [5, 10, 15, 20, 30, 40, 50]:
            args.k_recent = k_recent
            print("Sampling with {} recent neighbors".format(k_recent))
            for C in tqdm(C_list):
                args.C = C

                if args.device != "cpu":
                    torch.cuda.empty_cache()

                val_score, val_score_test_id, val_score_norm, val_score_norm_test_id, svr, svr_norm = eval(train_data, val_data, temporal_data, dataset, args)

                if val_score > best_val_score or val_score_test_id > best_val_score:
                    best_val_score = max(val_score, val_score_test_id)
                    best_clf = svr
                    best_params = (num_mlp_layers, k_recent, C)
                    print("Just Update Best model with params:{}, {}, {}".format(num_mlp_layers, k_recent, C))
                
                if val_score_norm > best_val_score or val_score_norm_test_id > best_val_score:
                    best_val_score = max(val_score_norm, val_score_norm_test_id)
                    best_clf = svr_norm
                    best_params = (num_mlp_layers, k_recent, C)
                    print("Just Update Best model with params: {}, {}, {}".format(num_mlp_layers, k_recent, C))

                print("Score for {} {} {}: ".format(num_mlp_layers, k_recent, C), val_score, val_score_test_id, val_score_norm, val_score_norm_test_id, sep = " ")

            # test_score, test_score_test_id, test_score_norm, test_score_norm_test_id, _, _ = eval(train_data, val_data, temporal_data, dataset,
            #                                                                                 args, "test", test_data)
            # print("Test Score for {} {}: ".format(k_recent, C), test_score, test_score_test_id, test_score_norm, test_score_norm_test_id)
    return best_clf, best_params, best_val_score

def fit_kernel(kernel, train_labels, C = 1.0):
    # C_list = np.logspace(-1, 2, 60)
    svr = SVR(kernel = "precomputed", cache_size = 16000, max_iter = 500000)
    multi_svr = MultiOutputRegressor(svr)
    multi_svr.fit(kernel, train_labels)

    return multi_svr

def evaluate_score(name, dataset, y_true, y_pred):
    evaluator = Evaluator(name)
    eval_metric = dataset.eval_metric

    input_dict = {
    "y_true": y_true,
    "y_pred": y_pred,
    "eval_metric": [eval_metric]
    }
    result_dict = evaluator.eval(input_dict)
    score = result_dict[eval_metric]

    return score

def eval_step(neighbor_loader, current_time, temporal_data, dataset, clf, args, norm = False):
    n = temporal_data.num_nodes
    
    n_test_id, test_labels = torch.load("./labels_all_time/{}.pt".format(current_time))
    test_gram_matrix = torch.load("./test_gram_all_time/{}_{}_{}.pt".format(current_time, args.k_recent, args.num_mlp_layers))
    # print(test_gram_matrix.shape)
    # print(test_gram_matrix[n_test_id])
    test_gram = test_gram_matrix  / (test_gram_matrix[test_gram_matrix != 0].min())
    test_gram_test_id = test_gram_matrix[n_test_id]
    test_gram_test_id /= test_gram_test_id[test_gram_test_id != 0].min()

    if norm:
        test_diag_list = torch.load("./test_gram_all_time/diag_list_{}_{}_{}.pt".format(current_time, args.k_recent, args.num_mlp_layers))
        train_diag_list = torch.load("./train_diag_list_{}_{}.pt".format(args.k_recent, args.num_mlp_layers))
        test_diag = test_diag_list[-1]
        train_diag = train_diag_list[-1][-1]

        # w/o test_id
        scale = test_diag[:, None] * train_diag[None, :]
        scale = torch.sqrt(scale)
        test_gram = (test_gram / scale).nan_to_num()

        # w test_id
        test_diag = test_diag[n_test_id]
        scale = test_diag[:, None] * train_diag[None, :]
        scale = torch.sqrt(scale)
        test_gram_test_id = (test_gram_test_id / scale).nan_to_num()
        
    ##### evaluate 
    pred_labels = clf.predict(test_gram)
    score = evaluate_score("tgbn-trade", dataset, test_labels.numpy(), pred_labels)

    pred_labels_only_test_id = clf.predict(test_gram_test_id)
    score_only_test_id = evaluate_score("tgbn-trade", dataset, test_labels[n_test_id].numpy(), pred_labels_only_test_id)
    
    # pred_labels_only_test_id = torch.from_numpy(pred_labels_only_test_id).softmax(dim = -1).numpy()

    return score, score_only_test_id

def eval(train_data, val_data, temporal_data, dataset, args, mode = "Validation", test_data = None):
    n = temporal_data.num_nodes
    neighbor_loader = LastNeighborLoader(n, size = args.k_recent)
    neighbor_loader.insert(train_data.src, train_data.dst)

    n_train_id, train_labels = torch.load("./labels_all_time/{}.pt".format(train_data.t.max().item()))
    train_gram_matrix = torch.load("./node_gram_{}_{}.pt".format(args.k_recent, args.num_mlp_layers))
    train_gram = torch.clone(train_gram_matrix[-1])
    train_gram /= (train_gram[train_gram != 0].min())

    ### new pipeline
    # train_node_emb = torch.load("./new_pipeline/train_node_emb_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
    # train_adj = torch.load("./new_pipeline/train_adj_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
    # train_gram_matrix = torch.load("./new_pipeline/train_gram_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
    # train_diag_list = torch.load("./new_pipeline/train_diag_list_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
    # multi_svr = fit_kernel(train_gram_matrix, train_labels.numpy(), C = args.C)
    ###
    ### new pipeline
    # diag = train_gram_matrix.diag()
    # scale = torch.sqrt(diag[:, None] * diag[None, :])
    # norm_gram = (train_gram_matrix / scale).nan_to_num()
    # multi_svr_norm = fit_kernel(norm_gram, train_labels.numpy(), C = args.C)
    ### 

    multi_svr = fit_kernel(train_gram, train_labels.numpy(), C = args.C)
    norm_gram = []
    for gram in train_gram_matrix:
        gram /= (gram[gram != 0].min())
        diag = gram.diag()
        scale = diag[:, None] * diag[None, :]
        scale = torch.sqrt(scale)
        norm_gram.append((gram / scale).nan_to_num())
    
    multi_svr_norm = fit_kernel(norm_gram[-1], train_labels.numpy(), C = args.C)

    total_score, total_score_test_id = 0.0, 0.0
    total_score_norm, total_score_norm_test_id = 0.0, 0.0

    for current_time in val_data.t.unique():
        current_time = current_time.item()

        cur_src, cur_dst = temporal_data.src[temporal_data.t == current_time], temporal_data.dst[temporal_data.t == current_time]
        neighbor_loader.insert(cur_src, cur_dst)

        if mode == "Validation":
            val_score, val_score_only_test_id = eval_step(neighbor_loader, current_time, temporal_data, dataset, multi_svr, args)
            val_score_norm, val_score_norm_only_test_id = eval_step(neighbor_loader, current_time, temporal_data, dataset, multi_svr_norm, args, norm = True)

            ### new pipeline
            # val_score, val_score_only_test_id, val_score_norm, val_score_norm_only_test_id = evaluate_timestep(temporal_data, current_time, neighbor_loader, multi_svr, 
            #                                                                                                    multi_svr_norm, train_node_emb, train_adj, train_diag_list, args, save_test_gram = True)
            ###

            total_score += val_score
            total_score_test_id += val_score_only_test_id
            total_score_norm += val_score_norm
            total_score_norm_test_id += val_score_norm_only_test_id

    if mode != "Validation":
        for current_time in test_data.t.unique():
            current_time = current_time.item()

            cur_src, cur_dst = temporal_data.src[temporal_data.t == current_time], temporal_data.dst[temporal_data.t == current_time]
            neighbor_loader.insert(cur_src, cur_dst)

            test_score, test_score_only_test_id = eval_step(neighbor_loader, current_time, temporal_data, dataset, multi_svr, args)
            test_score_norm, test_score_norm_only_test_id = eval_step(neighbor_loader, current_time, temporal_data, dataset, multi_svr_norm, args)

            total_score += test_score
            total_score_test_id += test_score_only_test_id
            total_score_norm += test_score_norm
            total_score_norm_test_id += test_score_norm_only_test_id

    num_ts = val_data.t.unique().shape[-1]
    if mode != "Validation":
        num_ts = test_data.t.shape[-1]

    total_score /= num_ts
    total_score_test_id /= num_ts
    total_score_norm /= num_ts
    total_score_norm_test_id /= num_ts

    return total_score, total_score_test_id, total_score_norm, total_score_norm_test_id, multi_svr, multi_svr_norm

##################### new pipeline

def node_emb(temporal_data, current_time, neighbor_loader, args):
    n = temporal_data.num_nodes
    coefs = args.alpha ** ((-torch.arange(1, args.time_dim + 1) + 1) / args.beta)
    nodes = torch.arange(n)
    node_embedding = torch.zeros((n, args.time_dim + temporal_data.msg.shape[-1]))

    n_id, a, e_id = neighbor_loader(nodes)
    _, node_idx = a

    t_emb = (current_time - temporal_data.t[e_id]).unsqueeze(-1) * coefs
    t_emb = torch.cos(t_emb)
    msg = temporal_data.msg[e_id]
    t_emb = torch.cat((t_emb, msg), dim = -1)

    node_embedding = node_embedding = torch_scatter.scatter_add(src = t_emb, index = node_idx.unsqueeze(-1).broadcast_to(node_idx.shape[0], args.time_dim + temporal_data.msg.shape[-1]), out = node_embedding,
                                                dim = 0)
    
    return node_embedding

def train(temporal_data, current_time, neighbor_loader, args, save_train = False):
    time_mask = torch.logical_and(temporal_data.t >= temporal_data.t.min(), temporal_data.t <= current_time)
    
    if args.time_window is not None:
        time_mask = torch.logical_and(temporal_data.t >= current_time - args.time_window, temporal_data.t <= current_time)

    train_node_emb = node_emb(temporal_data, current_time, neighbor_loader, args)

    n = temporal_data.num_nodes
    row, col = temporal_data.src[time_mask], temporal_data.dst[time_mask]
    train_adj = torch.sparse_coo_tensor(indices = torch.stack((row, col)), values = torch.ones(row.shape[-1]), size = (n, n))
    train_adj = train_adj.to_dense()
    train_adj = ((train_adj + train_adj.T) > 0).to(torch.long)

    train_gram_matrix, train_diag_list = t_gntk.get_diag_list(train_node_emb, train_adj, args, return_ntk = args.node_ntk)

    _, train_labels = torch.load("./labels_all_time/{}.pt".format(current_time))

    if save_train:
        torch.save(train_node_emb, "./new_pipeline/train_node_emb_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
        torch.save(train_adj, "./new_pipeline/train_adj_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
        torch.save(train_gram_matrix, "./new_pipeline/train_gram_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))
        torch.save(train_diag_list, "./new_pipeline/train_diag_list_{}_{}_{}.pt".format(args.time_window, args.k_recent, args.num_mlp_layers))

    return train_node_emb, train_adj, train_gram_matrix, train_diag_list, train_labels

def evaluate_timestep(temporal_data, current_time, neighbor_loader, svr, svr_norm, train_node_emb, train_adj, train_diag_list, args, save_test_gram = False):
    if svr is None:
        time_mask = torch.logical_and(temporal_data.t >= temporal_data.t.min(), temporal_data.t <= current_time)

        if args.time_window is not None:
            time_mask = torch.logical_and(temporal_data.t >= current_time - args.time_window, temporal_data.t <= current_time)

        eval_node_emb = node_emb(temporal_data, current_time, neighbor_loader, args)

        n = temporal_data.num_nodes
        row, col = temporal_data.src[time_mask], temporal_data.dst[time_mask]
        eval_adj = torch.sparse_coo_tensor(indices = torch.stack((row, col)), values = torch.ones(row.shape[-1]), size = (n, n))
        eval_adj = eval_adj.to_dense()
        eval_adj = ((eval_adj + eval_adj.T) > 0).to(torch.long)

        eval_diag_list = t_gntk.get_diag_list(eval_node_emb, eval_adj, args)

        test_gram_matrix = t_gntk.gntk(eval_node_emb, train_node_emb, eval_adj, train_adj, eval_diag_list, train_diag_list, args)

        diag = test_gram_matrix.diag()
        scale = diag[:, None] * diag[None, :]
        scale = torch.sqrt(scale)
        norm_test_gram = (test_gram_matrix / scale).nan_to_num()

        if save_test_gram:
            torch.save(test_gram_matrix, "./new_pipeline/test_gram_all_time/{}_{}_{}_{}.pt".format(current_time, args.time_window, args.k_recent, args.num_mlp_layers))
        
        return 0
    ####

    test_gram_matrix = torch.load("./new_pipeline/test_gram_all_time/{}_{}_{}_{}.pt".format(current_time, args.time_window, args.k_recent, args.num_mlp_layers))
    diag = test_gram_matrix.diag()
    scale = diag[:, None] * diag[None, :]
    scale = torch.sqrt(scale)
    norm_test_gram = (test_gram_matrix / scale).nan_to_num()

    n_test_id, test_labels = torch.load("./labels_all_time/{}.pt".format(current_time))

    scores = [] # (normal (no test id, test_id) then norm )

    pred_labels = svr.predict(test_gram_matrix)
    score = evaluate_score("tgbn-trade", dataset, test_labels.numpy(), pred_labels)
    scores.append(score)

    pred_labels_test_id = svr.predict(test_gram_matrix[n_test_id])
    score = evaluate_score("tgbn-trade", dataset, test_labels[n_test_id].numpy(), pred_labels_test_id)
    scores.append(score)

    pred_labels = svr_norm.predict(norm_test_gram)
    score = evaluate_score("tgbn-trade", dataset, test_labels.numpy(), pred_labels)
    scores.append(score)

    pred_labels_test_id = svr_norm.predict(norm_test_gram[n_test_id])
    score = evaluate_score("tgbn-trade", dataset, test_labels[n_test_id].numpy(), pred_labels_test_id)
    scores.append(score)
    
    return scores    

def create_clf(C):
    svr = SVR(kernel = "precomputed", cache_size = 16000, max_iter = 500000)
    multi_svr = MultiOutputRegressor(svr)

if __name__ == "__main__":
    # best_svr, best_params, best_val_score = search_best()
    # num_mlp_layers, k_recent, C = best_params
    
    # n_test_id, test_labels = torch.load("./labels_all_time/{}.pt".format(2010))
    # test_gram_matrix = torch.load("./test_gram_all_time/{}_{}_{}.pt".format(2010, args.k_recent, args.num_mlp_layers))
    # print(test_gram_matrix[n_test_id])

    # np.save("best_svr_3.npy", best_svr)
    # np.save("best_params_svr_3.npy", best_params)

    for num_mlp_layers in [1, 2, 3]:
        val_scores = []
        test_scores = []
        best_clf = np.load("./best_svr_{}.npy".format(num_mlp_layers), allow_pickle = True)
        best_clf = best_clf.item()
        _, k_recent, args.C = np.load("./best_params_svr_{}.npy".format(num_mlp_layers), allow_pickle = True)
        args.k_recent = int(k_recent)
        args.num_mlp_layers = num_mlp_layers

        total_score = 0.0
        total_score_w_test_id = 0.0
        num_t_s = val_data.t.unique().shape[-1]
        for t in val_data.t.unique():
            score, score_w_test_id = eval_step(1, t, temporal_data, dataset, best_clf, args)
            print(score, score_w_test_id, sep = " ")
            total_score += score
            total_score_w_test_id += score_w_test_id
            val_scores.append(score_w_test_id)

        total_score /= num_t_s
        total_score_w_test_id /= num_t_s
        print(total_score, total_score_w_test_id, sep = " ")
        print("\n")

        total_score = 0.0
        total_score_w_test_id = 0.0
        num_t_s = test_data.t.unique().shape[-1]
        for t in test_data.t.unique():
            score, score_w_test_id = eval_step(1, t, temporal_data, dataset, best_clf, args)
            print(score, score_w_test_id, sep = " ")
            total_score += score
            total_score_w_test_id += score_w_test_id
            test_scores.append(score_w_test_id)

        total_score /= num_t_s
        total_score_w_test_id /= num_t_s
        print(total_score, total_score_w_test_id, sep = " ")
        print("\n")

        val_scores = np.array(val_scores)
        test_scores = np.array(test_scores)

        mean_val = val_scores.mean()
        mean_test = test_scores.mean()
        std_val = val_scores.std()
        std_test = test_scores.std()

        print("Val {}; {}".format(mean_val, std_val))
        print("Test {}; {}".format(mean_test, std_test))
