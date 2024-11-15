import os
import torch
from torch_geometric.data import TemporalData
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
import pandas as pd
import numpy as np
import torch_scatter

data_dirs = ["./raw_data/infectious_ct1/infectious_ct1", 
             "./raw_data/dblp_ct1/dblp_ct1",
             "./raw_data/tumblr_ct1/tumblr_ct1",
             "./raw_data/facebook_ct1/facebook_ct1",
             "./raw_data/highschool_ct1/highschool_ct1"]

data_names = ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1", "highschool_ct1"]

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

def temporal_graph_from_TUds_split_graphs(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge):
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

def pre_kernel(temporal_graph, args):
    if args.encode_time == False:
        args.time_dim = 1

    data_loader = TemporalDataLoader(temporal_graph, batch_size = temporal_graph.num_edges // (args.num_sub_graphs - 1) if 
                                    temporal_graph.num_edges % (args.num_sub_graphs - 1) else (temporal_graph.num_edges // args.num_sub_graphs))
    
    nodes = torch.unique(torch.cat((temporal_graph.src, temporal_graph.dst)))
    n = nodes.shape[0]
    neighbor_loader = LastNeighborLoader(n, size = args.k_recent)

    adjs, adjs_cnt, node_embs = [], [], []

    temporal_snapshots_src = []
    temporal_snapshots_dst = []
    temporal_snapshots_times = []

    for idx, data in enumerate(data_loader):
        batch_src, batch_dst = data.src, data.dst

        if idx == 0:
            temporal_snapshots_src.append(batch_src.tolist())
            temporal_snapshots_dst.append(batch_dst.tolist())
            temporal_snapshots_times.append(data.t.tolist())
        else:
            temporal_snapshots_src.append(temporal_snapshots_src[-1] + batch_src.tolist())
            temporal_snapshots_dst.append(temporal_snapshots_dst[-1] + batch_dst.tolist())
            temporal_snapshots_times.append(temporal_snapshots_times[-1] + data.t.tolist())

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

    return temporal_snapshots_src, temporal_snapshots_dst, temporal_snapshots_times

def prep(data_dir, data_name):
    # data_dir = ""
    # data_name = ""
    num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge = readTUds(data_dir)
    temporal_graphs = temporal_graph_from_TUds(num_graphs, graphs_label, graphs_node, node_mapping, graphs_edge)

    torch.save(torch.tensor([graphs_label]).to(torch.float), "./{}_labels.pt".format(data_name))

    for idx, temporal_graph in enumerate(temporal_graphs):
        os.makedirs('./datasets/{}/{}'.format(data_name, idx), exist_ok=True)
        file_path = "./datasets/{}/{}".format(data_name, idx)

        torch.save(temporal_graph, file_path + "/data.pt")

        temporal_snapshots_src, temporal_snapshots_dst, temporal_snapshots_times = pre_kernel(temporal_graph, args)

        for i in range(args.num_sub_graphs):
            os.makedirs('./datasets/{}_{}/{}'.format(data_name, i, idx), exist_ok = True)
            file_path = './datasets/{}_{}/{}'.format(data_name, i, idx)

            snapshot = TemporalData(src = torch.Tensor(temporal_snapshots_src[i]).to(torch.long),
                                    dst = torch.Tensor(temporal_snapshots_dst[i]).to(torch.long),
                                    t = torch.Tensor(temporal_snapshots_times[i]).to(torch.long))
            
            torch.save(snapshot, file_path + "/data.pt")


if __name__ == "__main__":
    for i in range(len(data_dirs)):
        prep(data_dirs[i], data_names[i])
    
