import torch
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
import pandas as pd
import numpy as np
import torch_scatter
from temp_gntk import *
from torch_geometric.nn.models.tgn import LastNeighborLoader
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
# from networkx.classes.graph import Graph
import networkx

def readTUds(folder_name):
    """
        return temporal graphs
    """    
    num_graphs = 0
    graphs_label = []

    label_file = open(folder_name + "_graph_labels.txt")
    for line in label_file.readlines():
        num_graphs += 1
        graphs_label.append(int(line))

    graphs_node = [[] for _ in range(num_graphs)]
    node_mapping = {}
    indicator = open(folder_name + "_graph_indicator.txt")
    for idx, line in enumerate(indicator.readlines()):
        graph_id = int(line)
        graphs_node[graph_id - 1].append(idx + 1)
        if (idx + 1) not in node_mapping:
            node_mapping[idx + 1] = graph_id

    graphs_edge = [[] for _ in range(num_graphs)]
    check_edge = {}
    A = open(folder_name + "_A.txt")
    edge_t = open(folder_name + "_edge_attributes.txt")
    for line, t in zip(A.readlines(), edge_t.readlines()):
        u, v = line.split(", ")
        u, v = int(u), int(v)
        t = float(t)
        if (t, v, u) not in check_edge:
            graph_id = node_mapping[u]
            graphs_edge[graph_id - 1].append([t, u, v])
            check_edge[(t, v, u)] = 1
            check_edge[(t, u, v)] = 1

    return num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge

def temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge):
    for graph_id in range(num_graphs):
            graphs_edge[graph_id].sort()

    temporal_graphs = []

    for graph_id in range(num_graphs):
        src, dst, t_s = [], [], []
        for t, u, v in graphs_edge[graph_id]:
            src.append(u)
            dst.append(v)
            t_s.append(t)
                
        src = torch.Tensor(src).to(torch.long)
        dst = torch.Tensor(dst).to(torch.long)
        t_s = torch.Tensor(t_s).to(torch.long)

        min_id = torch.min(src.min(), dst.min())
        src -= min_id
        dst -= min_id

        temporal_graph = TemporalData(src = src, dst = dst, t = t_s)
        temporal_graphs.append(temporal_graph)

    return temporal_graphs

def pre_kernel(temporal_graph, current_time, args):
    # return adj within time window, node_embs

    if args.time_window is None:
        time_mask = torch.logical_and()

def get_diag_gram_matrix(ds, args):
    t_gntk = TemporalGNTK()
    
    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = ds
    diag_gram_matrix = [[] for _ in range(args.num_sub_graphs)]

    for i in range(args.num_sub_graphs):
        for j in range(len(graphs_adjs)):
            diag_list, adj, node_emb = graphs_diag_list[j][i], graphs_adjs[j][i], graphs_node_embs[j][i]
            gntk_val = t_gntk.gntk(node_emb, node_emb, adj, adj, diag_list, diag_list, args)
            diag_gram_matrix[i].append(gntk_val)

    return np.array(diag_gram_matrix)

def normalize_gram(gram_matrix, args, mode = "train", diag_list_train = None, diag_list_test = None):
    if mode == "test":
        normalized_gram = []
        for i in range(args.num_sub_graphs):
            temporal_gram = np.copy(gram_matrix[:, :, i])
            diag_test, diag_train = diag_list_test[i], diag_list_train[i]
            temporal_gram /= (diag_test[:, None] * diag_train[None, :])

            normalized_gram.append(temporal_gram)
        
        return np.array(normalized_gram)

    normalized_gram = []
    for subgraph_id in range(args.num_sub_graphs):
        temporal_gram = np.copy(gram_matrix[:, :, subgraph_id])
        diag = np.sqrt(np.diag(temporal_gram))
        temporal_gram /= (diag[:, None] * diag[None, :])

        normalized_gram.append(temporal_gram)
    
    return np.array(normalized_gram)

def search(kernel, labels, C_list):
    svc = SVC(kernel = "precomputed", cache_size = 16000, max_iter = 500000)
    clf = GridSearchCV(svc, {'C' : C_list}, 
                    n_jobs=80, verbose=0, return_train_score=True)
    clf.fit(kernel, labels)

    df = pd.DataFrame({'C': C_list, 
                       'train': clf.cv_results_['mean_train_score'], 
                       'test': clf.cv_results_['mean_test_score']}, 
                        columns=['C', 'train', 'test'])
    return df['test'].max(), df['train'].max(), clf

def pre_kernel(temporal_graph, args):
    if args.encode_time == False:
        args.time_dim = 1

    data_loader = TemporalDataLoader(temporal_graph, batch_size = temporal_graph.num_edges // (args.num_sub_graphs - 1) if 
                                    temporal_graph.num_edges % (args.num_sub_graphs - 1) else (temporal_graph.num_edges // args.num_sub_graphs))
    
    nodes = torch.unique(torch.cat((temporal_graph.src, temporal_graph.dst)))
    n = nodes.shape[0]
    neighbor_loader = LastNeighborLoader(n, size = args.k_recent)

    adjs, adjs_cnt, node_embs = [], [], []

    for idx, data in enumerate(data_loader):
        batch_src, batch_dst = data.src, data.dst
        
        if idx == 0:
            adj_cnt = torch.zeros((n, n))
        else:
            adj_cnt = torch.clone(adjs_cnt[-1])

        for u, v in zip(batch_src, batch_dst):
            adj_cnt[u.item()][v.item()] += 1
            adj_cnt[v.item()][u.item()] += 1
        adjs_cnt.append(adj_cnt)

        adj = (adj_cnt > 0).to(torch.long)
        adjs.append(adj)

        current_time = data.t.max()
        neighbor_loader.insert(batch_src, batch_dst)

        n_id, a, e_id = neighbor_loader(nodes)
        _, node_idx = a
        node_embedding = torch.zeros((n, args.time_dim)) # [N, d]
        deg = torch.zeros(nodes.shape) # [N]

        t_s = temporal_graph.t[e_id]
        if args.relative_difference:
            t_s = current_time - temporal_graph.t[e_id]
        
        if args.encode_time:
            t_emb = t_s.unsqueeze(-1) * (args.alpha ** ((-torch.arange(1, args.time_dim + 1) + 1) / args.beta))
            t_emb = torch.cos(t_emb)
        else:
            t_emb = t_s.unsqueeze(-1).to(node_embedding.dtype)


        # normal t_emb
        # t_emb = (current_time - temporal_graph.t[e_id]).unsqueeze(-1) * (args.alpha ** ((-torch.arange(1, args.time_dim + 1) + 1) / args.beta))
        # t_emb = torch.cos(t_emb)
        
        # relative difference
        # t_emb = (current_time - temporal_graph.t[e_id]).unsqueeze(-1).to(node_embedding.dtype)

        # absolute time
        # t_emb = (temporal_graph.t[e_id]).unsqueeze(-1).to(node_embedding.dtype)

        # absolute time enc
        # t_emb = (temporal_graph.t[e_id]).unsqueeze(-1) * (args.alpha ** ((-torch.arange(1, args.time_dim + 1) + 1) / args.beta))
        # t_emb = torch.cos(t_emb)

        # deg = torch_scatter.scatter_add(src = torch.ones(node_idx.shape), index = node_idx, out = deg)
        deg = adj.sum(dim = -1)
        deg += (deg == 0)

        # [N, time_dim]
        node_embedding = torch_scatter.scatter_add(src = t_emb, index = node_idx.unsqueeze(-1).broadcast_to(node_idx.shape[0], args.time_dim), out = node_embedding,
                                                dim = 0)
        if args.neighborhood_avg:
            node_embedding /= deg.unsqueeze(-1)

        if args.node_onehot:
            one_hot_emb = torch.eye(n)
            node_embedding = torch.cat([node_embedding, one_hot_emb], dim = -1)
        
        node_embs.append(node_embedding)

    return adjs, adjs_cnt, node_embs

def pre_t_gntk(train_ds, args):
    t_gntk = TemporalGNTK()
    graphs_diag_list = []
    graphs_adjs, graphs_adjs_cnt, graphs_node_embs = [], [], []
    
    for graph in train_ds:
        temporal_graph_diag_list = []

        adjs, adjs_cnt, node_embs = pre_kernel(graph, args)    
        
        for adj, adj_cnt,node_emb in zip(adjs, adjs_cnt, node_embs):
            diag_list = t_gntk.get_diag_list(node_emb, adj, args)
            temporal_graph_diag_list.append(diag_list)

        graphs_diag_list.append(temporal_graph_diag_list)
        graphs_adjs.append(adjs)
        graphs_adjs_cnt.append(adjs_cnt)
        graphs_node_embs.append(node_embs)

    return graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs

def compute_gram_matrix_from_graph_embs(graph_embs, mode = "train", test_graph_embs = None, normalize = False):
    if mode == "test":
        if normalize:
            scale_test = (test_graph_embs ** 2).sum(axis = -1).reshape(test_graph_embs.shape[0], 1)
            scale_train = (graph_embs ** 2).sum(axis = -1).reshape(1, graph_embs.shape[0])
            scale = scale_test.dot(scale_train)
            scale = np.sqrt(scale)
            return test_graph_embs.dot(graph_embs.T) / scale
    
        return test_graph_embs.dot(graph_embs.T)
    
    if normalize:
        scale = (graph_embs ** 2).sum(axis = -1).reshape(graph_embs.shape[0], 1)
        scale = np.sqrt(scale)
        return graph_embs.dot(graph_embs.T) / (scale.dot(scale.T))
    
    return graph_embs.dot(graph_embs.T)

def compute_gram_matrix(train_ds, args, mode = "train", test_ds = None):
    t_gntk = TemporalGNTK()

    graphs_diag_list, graphs_adjs, graphs_adjs_cnt, graphs_node_embs = train_ds
    n = len(graphs_adjs)

    if mode == "test":
        test_graphs_diag_list, test_graphs_adjs, _, test_graphs_node_embs = test_ds
        n_test = len(test_graphs_adjs)
        test_gram_matrix = [[[] for _ in range(n)] for _ in range(n_test)]

        for i in range(n_test):
            for j in range(n):
                for k in range(args.num_sub_graphs):
                    node_emb_test, node_emb_train = test_graphs_node_embs[i][k], graphs_node_embs[j][k]
                    A_test, A_train = test_graphs_adjs[i][k], graphs_adjs[j][k]
                    diag_list_test, diag_list_train = test_graphs_diag_list[i][k], graphs_diag_list[j][k]

                    gntk_val = t_gntk.gntk(node_emb_test, node_emb_train, A_test, A_train, diag_list_test, diag_list_train, args)
                    test_gram_matrix[i][j].append(gntk_val)

        return np.array(test_gram_matrix)

    gram_matrix = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            gram_matrix[i][j] = []

            for k in range(args.num_sub_graphs):
                node_emb_1, node_emb_2 = graphs_node_embs[i][k], graphs_node_embs[j][k]
                A_1, A_2 = graphs_adjs[i][k], graphs_adjs[j][k]
                diag_list_1, diag_list_2 = graphs_diag_list[i][k], graphs_diag_list[j][k]

                gntk_val = t_gntk.gntk(node_emb_1.to("cuda"), node_emb_2.to("cuda"), A_1.to("cuda"), A_2.to("cuda"), diag_list_1, diag_list_2, args)
                gntk_val = gntk_val.detach().cpu()
                gram_matrix[i][j].append(gntk_val)

    for i in range(n):
        for j in range(i):
            gram_matrix[i][j] = gram_matrix[j][i]

    return np.array(gram_matrix)

def adj_to_nxgraph(adj):
    G = networkx.classes.graph.Graph()
    n = adj.shape[0]
    nodes_list = torch.arange(n)
    edges_list = []

    for i in range(n):
        for j in range(n):
            if adj[i][j]:
                edges_list.append((i, j))

    G.add_nodes_from(nodes_list.tolist())
    G.add_edges_from(edges_list)

    return G

def grakel_graphs(ds, args, get_label = True, num_slice = 5):
    graphs_adjs, graphs_adjs_cnt, graphs_node_embs = ds
    
    grakel_graphs_adjs = []
    grakel_node_attrs = []

    for graph_adjs in graphs_adjs:
        adj = graph_adjs[num_slice - 1]
        adj = adj.numpy()
        grakel_graphs_adjs.append(adj.tolist())

    if get_label:
        for dim in range(args.time_dim):
            grakel_node_attrs_per_dim = []
            for graph_node_embs in graphs_node_embs:
                graph_node_emb = graph_node_embs[num_slice - 1]
                graph_node_emb = graph_node_emb.numpy()
                n = graph_node_emb.shape[0]

                graph_node_attr = {}
                for node_id in range(n):
                    graph_node_attr[node_id] = graph_node_emb[node_id].tolist()[dim]
                    
                grakel_node_attrs_per_dim.append(graph_node_attr)
            
            grakel_node_attrs.append(grakel_node_attrs_per_dim)
    else:
        for graph_node_embs in graphs_node_embs:
            graph_node_emb = graph_node_embs[-1].numpy()
            n = graph_node_emb.shape[0]

            graph_node_attr = {}
            for node_id in range(n):
                graph_node_attr[node_id] = graph_node_emb[node_id].tolist()
            
            grakel_node_attrs.append(graph_node_attr)

    return grakel_graphs_adjs, grakel_node_attrs
