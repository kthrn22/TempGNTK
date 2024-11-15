import time
import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader

from torch_geometric.nn.models.tgn import LastNeighborLoader
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.data import TemporalData
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
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]
    
    def __getitem__(self, idx):
        return {"X": self.X[idx],
         "y": self.y[idx]}
    
class Trainer(object):
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_step(self, train_loader):
        # need loss and precision
        
        self.model.train()
        avg_train_loss = 0.0
        avg_train_acc = 0.0

        for i, batch in enumerate(train_loader):
            graphs_emb, graphs_label = batch["X"].to(self.device), batch["y"].to(self.device)

            self.optimizer.zero_grad()
            predict_labels = self.model(graphs_emb)
            train_loss = self.criterion(predict_labels, graphs_label).mean()
            train_loss.backward()
            self.optimizer.step()
            
            train_acc = torch.sum((predict_labels > 0.5) == graphs_label) / graphs_label.shape[0]

            avg_train_loss += train_loss.detach().item()
            avg_train_acc += train_acc 

        avg_train_loss /= len(train_loader)
        avg_train_acc /= len(train_loader)
            
        print("Avg Train Loss: {}".format(avg_train_loss))
        print("Avg Train Acc: {}".format(avg_train_acc))

        return avg_train_acc, avg_train_loss

    def eval_step(self, valid_loader):
        self.model.eval()

        avg_val_loss = 0.0
        avg_val_acc = 0.0

        with torch.inference_mode():
            for i, batch in enumerate(valid_loader):
                graphs_emb, graphs_label = batch["X"].to(self.device), batch["y"].to(self.device)

                predict_labels = self.model(graphs_emb)
                val_loss = self.criterion(predict_labels, graphs_label).mean()

                val_acc = torch.sum((predict_labels > 0.5) == graphs_label) / graphs_label.shape[0]

                avg_val_loss += val_loss.item()
                avg_val_acc += val_acc
            
            avg_val_loss /= len(valid_loader)
            avg_val_acc /= len(valid_loader)

        print("Avg Val Loss: {}".format(avg_val_loss))
        print("Avg Val Acc: {}".format(avg_val_acc))

        return avg_val_acc, avg_val_loss

    def train(self, num_epochs, train_loader, valid_loader):
        best_val_acc = 0.0

        for epoch in range(1, num_epochs + 1):
            print("Epoch: {}".format(epoch))

            print("Training ...")
            pbar = tqdm(total = len(train_loader))
            avg_train_acc, avg_train_loss = self.train_step(train_loader)

            print("Validating ...")
            pbar = tqdm(total = len(valid_loader))
            avg_val_acc, avg_val_loss = self.eval_step(valid_loader)

            if best_val_acc < avg_val_acc:
                best_val_acc = avg_val_acc

            print("----------------")

        return best_val_acc
    
def run(data_name, len_data, all_graph):
    device = "cpu"
    num_epochs = 10
    graphs_emb = torch.load("./{}_graphs_emb.pt".format(data_name)).view(len_data, -1)
    graphs_label = torch.load("./{}_labels.pt".format(all_graph)).view(len_data, -1)

    start_time = time.time()

    k_fold_score = 0.0
    k_fold_scores = []

    ds = Dataset(graphs_emb, graphs_label)
    n = len(ds)
    
    split_ids = [((i * n) // 5) for i in range(1, 5)] 
    
    for k_fold in range(5):
        if k_fold == 0:
            train_ds = ds[split_ids[0]:]
            val_ds = ds[:split_ids[0]]
        elif k_fold == 4:
            train_ds = ds[:split_ids[-1]]
            val_ds = ds[split_ids[-1]:]
        else:
            X = torch.cat([ds[:split_ids[k_fold - 1]]["X"], ds[split_ids[k_fold]:]["X"]])
            y = torch.cat([ds[:split_ids[k_fold - 1]]["y"], ds[split_ids[k_fold]:]["y"]])
            train_ds = {"X": X, "y": y}
            val_ds = ds[split_ids[k_fold - 1] : split_ids[k_fold]]       

        train_ds = Dataset(train_ds["X"], train_ds["y"])
        val_ds = Dataset(val_ds["X"], val_ds["y"])
        
        train_loader = DataLoader(train_ds, batch_size = len(train_ds) // 5, shuffle = True)
        valid_loader = DataLoader(val_ds, batch_size = len(val_ds) // 5, shuffle = True)

        model = GraphClassifier(graphs_emb.shape[-1]).to('cpu')
        optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
        criterion = nn.BCEWithLogitsLoss(reduction = 'none')

        trainer = Trainer(model, optimizer, criterion, device)
        fold_score = trainer.train(num_epochs, train_loader, valid_loader)
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

if __name__ == "__main__":
    data_names = ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1", "highschool_ct1"]
    # data_name = "tumblr_ct1"
    lens = [200, 755, 373, 995, 180]
    # all_info = []
    # for data_name, num_graphs in zip(data_names, lens):
    #     score, std, total_time = run(data_name, num_graphs)
    #     all_info.append((score, std, total_time))

    all_results = {}

    for data_name, len_data in zip(["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1"], [200, 755, 373, 995]):
        
        all_results[data_name] = []
        
        for i in range(4):
            score, std, total_time = run(data_name + "_{}".format(i), len_data, data_name)
            all_info = (score, std, total_time)

            all_results[data_name].append(all_info)
    
    print(all_results)
