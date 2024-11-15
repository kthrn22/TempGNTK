from tqdm import tqdm
import pdb
import time
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader
from models.evovlegcnh import EvolveGCNH
from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
# from torch_geometric_temporal import EvolveGCNH
# from temp_gntk import *
# import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import pandas as pd
import torch_scatter
# from utils_graph_classification import *
# from grakel.kernels import WeisfeilerLehman, ShortestPathAttr, ShortestPath, RandomWalkLabeled
# from grakel.graph import Graph
from tqdm import tqdm

def get_all_snapshots(src, dst, ts):
    # define edge_index
    # define node_features
    # get all for one graph
    num_nodes = 100
    # one hot node features
    node_feat = torch.eye((num_nodes))
    node_feat = node_feat[:, :32] * 0

    all_snapshots_edge_index = []
    for t in ts.unique():
        mask = ts == t

        src_ids, dst_ids = src[mask], dst[mask]
        edge_index = torch.cat([src_ids.unsqueeze(0), dst_ids.unsqueeze(0)], dim = 0)

        all_snapshots_edge_index.append(edge_index)

    return (node_feat, all_snapshots_edge_index)

class GraphEmbedder(nn.Module):
    # get embedding for whole graph
    def __init__(self, num_nodes, in_channels, num_layers):
        super(GraphEmbedder, self).__init__()
        self.layers = nn.ModuleList([
            EvolveGCNH(num_of_nodes = num_nodes, in_channels = in_channels)  for _ in range(num_layers)])

    def forward(self, raw_node_feat, all_snapshots_edge_index):
        graph_emb = torch.zeros(raw_node_feat.shape[-1]).to(raw_node_feat.device)
        
        for edge_index in all_snapshots_edge_index:
            node_feat = raw_node_feat
            for layer in self.layers:
                node_feat = layer(node_feat, edge_index.to(raw_node_feat.device))

            graph_emb += (node_feat.sum(dim = 0))    

        return graph_emb

class GraphClassifier(nn.Module):
    def __init__(self, in_dim):
        super(GraphClassifier, self).__init__()
        self.in_dim = in_dim

        self.lin = nn.Linear(in_dim, in_dim)
        self.out_fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.lin(x)
        x = torch.nn.functional.relu(x)
        x = self.out_fc(x)
        x = torch.sigmoid(x)

        return x
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, temporal_graphs, labels):
        self.temporal_graphs = temporal_graphs
        self.labels = labels

    def __len__(self):
        return len(self.temporal_graphs)
    
    def __getitem__(self, idx):
        return {"src": [temporal_graph.src for temporal_graph in self.temporal_graphs[idx]],
                "dst": [temporal_graph.dst for temporal_graph in self.temporal_graphs[idx]],
                "t": [temporal_graph.t for temporal_graph in self.temporal_graphs[idx]],
                "y": self.labels[idx]}

class Trainer(object):
    def __init__(self, model, optimizer, criterion, device):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, train_ds, train_bs):
        # need loss and precision
        
        self.model.train()
        avg_train_loss = 0.0
        avg_train_acc = 0.0
        # pbar = tqdm(total = len(train_ds))
        for i in tqdm(range(len(train_ds))):
            # get graphs_emb
            batch_graph_emb = torch.tensor([]).to(self.device)
            batch_temporal_graphs = train_ds[i: min(len(train_ds), i + train_bs)]

            graph_src_list, graph_dst_list, graph_t_list = batch_temporal_graphs["src"], batch_temporal_graphs["dst"], batch_temporal_graphs["t"]

            for src, dst, ts in zip(graph_src_list, graph_dst_list, graph_t_list):
                node_feat, all_snapshots_edge_index = get_all_snapshots(src, dst, ts)
                
                graph_emb = self.model[0](node_feat.to(self.device), all_snapshots_edge_index)
                batch_graph_emb = torch.cat([batch_graph_emb, graph_emb.unsqueeze(0)], dim = 0)

            graphs_label = batch_temporal_graphs['y'].to(self.device)

            # graphs_emb, graphs_label = batch["X"].to(self.device), batch["y"].to(self.device)

            self.optimizer.zero_grad()
            predict_labels = self.model[1](batch_graph_emb)
            # pdb.set_trace()
            if len(predict_labels.shape) > 0:
                predict_labels = predict_labels.flatten()
            if len(graphs_label.shape) > 0:
                graphs_label = graphs_label.flatten()
            train_loss = self.criterion(predict_labels, graphs_label).mean()
            
            train_loss.backward()
            self.optimizer.step()
            
            train_acc = torch.sum((predict_labels > 0.5) == graphs_label) / graphs_label.shape[0]

            avg_train_loss += train_loss.detach().item()
            avg_train_acc += train_acc 

        avg_train_loss /= len(train_ds)
        avg_train_acc /= len(train_ds)
            
        print("Avg Train Loss: {}".format(avg_train_loss))
        print("Avg Train Acc: {}".format(avg_train_acc))

        return avg_train_acc, avg_train_loss

    def eval_step(self, val_ds, val_bs):
        self.model.eval()

        avg_val_loss = 0.0
        avg_val_acc = 0.0

        with torch.inference_mode():
            for i in tqdm(range(len(val_ds))):
                batch_graph_emb = torch.tensor([]).to(self.device)
                batch_temporal_graphs = val_ds[i: min(len(val_ds), i + val_bs)]

                graph_src_list, graph_dst_list, graph_t_list = batch_temporal_graphs["src"], batch_temporal_graphs["dst"], batch_temporal_graphs["t"]
                for src, dst, ts in zip(graph_src_list, graph_dst_list, graph_t_list):
                    node_feat, all_snapshots_edge_index = get_all_snapshots(src, dst, ts)
                    graph_emb = self.model[0](node_feat.to(self.device), all_snapshots_edge_index)

                    batch_graph_emb = torch.cat([batch_graph_emb, graph_emb.unsqueeze(0)], dim = 0)

                graphs_label = batch_temporal_graphs['y'].to(self.device)
                
                predict_labels = self.model[1](batch_graph_emb)
                
                if len(predict_labels.shape) > 0:
                    predict_labels = predict_labels.flatten()
                if len(graphs_label.shape) > 0:
                    graphs_label = graphs_label.flatten()

                val_loss = self.criterion(predict_labels, graphs_label).mean()

                val_acc = torch.sum((predict_labels > 0.5) == graphs_label) / graphs_label.shape[0]

                avg_val_loss += val_loss.item()
                avg_val_acc += val_acc
            
            avg_val_loss /= len(val_ds)
            avg_val_acc /= len(val_ds)

        print("Avg Val Loss: {}".format(avg_val_loss))
        print("Avg Val Acc: {}".format(avg_val_acc))

        return avg_val_acc, avg_val_loss

    def train(self, num_epochs, train_ds, train_bs, val_ds, val_bs):
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print("Epoch: {}".format(epoch))

            print("Training ...")
            pbar = tqdm(total = len(train_ds))
            avg_train_acc, avg_train_loss = self.train_step(train_ds, train_bs)

            print("Validating ...")
            pbar = tqdm(total = len(val_ds))
            avg_val_acc, avg_val_loss = self.eval_step(val_ds, val_bs)

            if best_val_acc < avg_val_acc:
                best_val_acc = avg_val_acc

            print("----------------")

        return best_val_acc
    
def run(data_name, len_data, graph_name):
    device = "cpu"
    num_epochs = 10

    temporal_graphs = []
    for i in range(len_data):
        temporal_graphs.append(torch.load(f'./datasets/{graph_name}/{i}/data.pt'))

    # graphs_emb = torch.load("./{}_graphs.pt".format(data_name)).view(len_data, -1)
    graphs_label = torch.load("./{}_labels.pt".format(graph_name)).view(len_data, -1)

    ds = Dataset(temporal_graphs, graphs_label)
    # pdb.set_trace()
    # return

    start_time = time.time()

    k_fold_score = 0.0
    k_fold_scores = []

    n = len(ds)
    
    split_ids = [((i * n) // 5) for i in range(1, 5)] 
    
    for k_fold in range(5):
        if k_fold == 0:
            continue
        if k_fold == 0:
            train_ds = Dataset(temporal_graphs[split_ids[0]:], graphs_label[split_ids[0]:])
            val_ds = Dataset(temporal_graphs[:split_ids[0]], graphs_label[:split_ids[0]])
            # train_ds = ds[split_ids[0]:]
            # val_ds = ds[:split_ids[0]]
        elif k_fold == 4:
            train_ds = Dataset(temporal_graphs[:split_ids[-1]], graphs_label[:split_ids[-1]])
            val_ds = Dataset(temporal_graphs[split_ids[-1]:], graphs_label[split_ids[-1]:])
            
            # train_ds = ds[:split_ids[-1]]
            # val_ds = ds[split_ids[-1]:]
        else:
            # src = torch.cat([ds[:split_ids[k_fold - 1]]["src"], ds[split_ids[k_fold]:]["src"]])
            # dst = torch.cat([ds[:split_ids[k_fold - 1]]["dst"], ds[split_ids[k_fold]:]["dst"]])
            # t = torch.cat([ds[:split_ids[k_fold - 1]]["t"], ds[split_ids[k_fold]:]["t"]])
            # y = torch.cat([ds[:split_ids[k_fold - 1]]["y"], ds[split_ids[k_fold]:]["y"]])
            # train_ds = {"X": X, "y": y}

            train_ds = Dataset(temporal_graphs[:split_ids[k_fold - 1]] + temporal_graphs[split_ids[k_fold]:], 
                               torch.tensor(graphs_label.flatten().tolist()[:split_ids[k_fold - 1]] + graphs_label.flatten().tolist()[split_ids[k_fold]:]))
            
            val_ds = Dataset(temporal_graphs[split_ids[k_fold - 1] : split_ids[k_fold]], 
                               graphs_label[split_ids[k_fold - 1] : split_ids[k_fold]])
            # pdb.set_trace()

            # val_ds = ds[split_ids[k_fold - 1] : split_ids[k_fold]]       

        # train_ds = Dataset(train_ds["X"], train_ds["y"])
        # val_ds = Dataset(val_ds["X"], val_ds["y"])
        
        # train_loader = DataLoader(train_ds, batch_size = len(train_ds) // 5, shuffle = True)
        # valid_loader = DataLoader(val_ds, batch_size = len(val_ds) // 5, shuffle = True)

        model = nn.Sequential(GraphEmbedder(100, 32, 1),
                            GraphClassifier(32)).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')

        trainer = Trainer(model, optimizer, criterion, device)

        fold_score = trainer.train(num_epochs, train_ds, len(train_ds) // 5, val_ds, len(val_ds) // 5)
        fold_score = fold_score.detach().cpu()
        print("Fold {} score: {}".format(k_fold + 1, fold_score))

        # k_fold_score += fold_score
        k_fold_scores.append(fold_score)
    
    k_fold_scores = np.array(k_fold_scores)
    avg_fold_score = np.sum(k_fold_scores) / 5

    std = np.sqrt(np.sum((k_fold_scores - avg_fold_score) ** 2) / 5)

    total_time = time.time() - start_time

    print("Average Fold Score: {}".format(avg_fold_score))
    print("Std: {}".format(std))
    return avg_fold_score, std, total_time
    # print(f'Total time {data_name}: {total_time:.2f} seconds')


if __name__ == "__main__":    
    # # if args.graphs_dataset.startswith("dblp"):
    # #     num_graphs = 755
    # # if args.graphs_dataset.startswith("facebook"):
    # #     num_graphs = 995
    # # if args.graphs_dataset.startswith("tumblr"):
    # #     num_graphs = 373
    # # if args.graphs_dataset.startswith("infectious"):
    # #     num_graphs = 200
    # # data_names = ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1", "highschool_ct1"]
    # # # data_name = "tumblr_ct1"
    # # lens = [200, 755, 373, 995, 180]
    # # # all_info = []
    # # # for data_name, num_graphs in zip(data_names, lens):
    # # #     score, std, total_time = run(data_name, num_graphs)
    # # #     all_info.append((score, std, total_time))

    # # score, std, total_time = run(data_names[-1], lens[-1])
    # # all_info = (score, std, total_time)
    # # print(all_info)

    # all_results = {}

    for data_name, len_data in zip(["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1"], [200, 755, 373, 995]):
    # for data_name, len_data in zip(["dblp_ct1"], [755]):
        # all_results[data_name] = []

        avg_fold_score, std, total_time = run(data_name, len_data, data_name)
        print(avg_fold_score, std, total_time)
        # for i in range(4):
        #     score, std, total_time = run(data_name + "_{}".format(i), len_data, data_name)
        #     all_info = (score, std, total_time)

            # all_results[data_name].append(all_info)


    
    # print(all_results)

