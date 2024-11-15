# This code achieves a performance of around 96.60%. However, it is not
# directly comparable to the results reported by the TGN paper since a
# slightly different evaluation setup is used here.
# In particular, predictions in the same batch are made in parallel, i.e.
# predictions for interactions later in the batch have no access to any
# information whatsoever about previous interactions in the same batch.
# On the contrary, when sampling node neighborhoods for interactions later in
# the batch, the TGN paper code has access to previous interactions in the
# batch.
# While both approaches are correct, together with the authors of the paper we
# decided to present this version here as it is more realsitic and a better
# test bed for future methods.

### TGN impplementation from: https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py

import os.path as osp
import time
import argparse
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn import Linear
from torch.utils.data import ConcatDataset
from torch_geometric.datasets import JODIEDataset
from torch_geometric.loader import TemporalDataLoader
from torch_geometric.nn import TGNMemory, TransformerConv
# from torch_geometric.nn.models.tgn import (
#     IdentityMessage,
#     LastAggregator,
#     LastNeighborLoader,
# )

from tgn_model import *
from torch_geometric.data import TemporalData
from tqdm import tqdm

device = "cpu"

# path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'JODIE')
# dataset = JODIEDataset(path, name='wikipedia')
# data = dataset[0]

# For small datasets, we can put the whole dataset on GPU and thus avoid
# expensive memory transfer costs for mini-batches:

memory_dim = time_dim = embedding_dim = 100
num_epochs = 10

# data = torch.load("./datasets/dblp_ct1/0/data.pt")
# msg = torch.zeros((data.t.shape)).view(data.t.shape[0], -1)
# data = TemporalData(src = data.src, dst = data.dst, t = data.t, msg = msg)
# data = data.to(device)

# train_data, val_data, test_data = data.train_val_test_split(
#     val_ratio=0.15, test_ratio=0.15)

# train_loader = TemporalDataLoader(
#     train_data,
#     batch_size=200,
#     neg_sampling_ratio=1.0,
# )
# val_loader = TemporalDataLoader(
#     val_data,
#     batch_size=200,
#     neg_sampling_ratio=1.0,
# )
# test_loader = TemporalDataLoader(
#     test_data,
#     batch_size=200,
#     neg_sampling_ratio=1.0,
# )
# neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

class GraphAttentionEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        self.conv = TransformerConv(in_channels, out_channels // 2, heads=2,
                                    dropout=0.1, edge_dim=edge_dim)

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        return self.conv(x, edge_index, edge_attr)


class LinkPredictor(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.lin_src = Linear(in_channels, in_channels)
        self.lin_dst = Linear(in_channels, in_channels)
        self.lin_final = Linear(in_channels, 1)

    def forward(self, z_src, z_dst):
        h = self.lin_src(z_src) + self.lin_dst(z_dst)
        h = h.relu()
        return self.lin_final(h)

# memory = TGNMemory(
#     data.num_nodes,
#     data.msg.size(-1),
#     memory_dim,
#     time_dim,
#     message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
#     aggregator_module=LastAggregator(),
# ).to(device)

# gnn = GraphAttentionEmbedding(
#     in_channels=memory_dim,
#     out_channels=embedding_dim,
#     msg_dim=data.msg.size(-1),
#     time_enc=memory.time_enc,
# ).to(device)

# link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

# optimizer = torch.optim.Adam(
#     set(memory.parameters()) | set(gnn.parameters())
#     | set(link_pred.parameters()), lr=0.0001)
# criterion = torch.nn.BCEWithLogitsLoss()

# Helper vector to map global node indices to local ones.
# assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)


def train(train_data, gnn, memory, link_pred, train_loader, neighbor_loader, assoc, criterion, optimizer):
    memory.train()
    gnn.train()
    link_pred.train()

    memory.reset_state()  # Start with a fresh memory.
    neighbor_loader.reset_state()  # Start with an empty graph.

    total_loss = 0
    node_emb = None
    pbar = tqdm(total = len(train_loader))
    for batch in train_loader:
        optimizer.zero_grad()
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        # Get updated memory of all nodes involved in the computation.
        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))
        
        node_emb = z

        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        loss = criterion(pos_out, torch.ones_like(pos_out))
        loss += criterion(neg_out, torch.zeros_like(neg_out))

        # Update memory and neighbor loader with ground-truth state.
        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)

        loss.backward()
        optimizer.step()
        memory.detach()
        total_loss += float(loss) * batch.num_events

    return total_loss / train_data.num_events, node_emb


@torch.no_grad()
def test(gnn, memory, link_pred, assoc, neighbor_loader, loader):
    memory.eval()
    gnn.eval()
    link_pred.eval()

    torch.manual_seed(12345)  # Ensure deterministic sampling across epochs.

    aps, aucs = [], []
    node_emb = None
    pbar = tqdm(total = len(loader))
    
    for batch in loader:
        batch = batch.to(device)

        n_id, edge_index, e_id = neighbor_loader(batch.n_id)
        assoc[n_id] = torch.arange(n_id.size(0), device=device)

        z, last_update = memory(n_id)
        z = gnn(z, last_update, edge_index, data.t[e_id].to(device),
                data.msg[e_id].to(device))

        node_emb = z

        pos_out = link_pred(z[assoc[batch.src]], z[assoc[batch.dst]])
        neg_out = link_pred(z[assoc[batch.src]], z[assoc[batch.neg_dst]])

        y_pred = torch.cat([pos_out, neg_out], dim=0).sigmoid().cpu()
        y_true = torch.cat(
            [torch.ones(pos_out.size(0)),
             torch.zeros(neg_out.size(0))], dim=0)

        aps.append(average_precision_score(y_true, y_pred))
        aucs.append(roc_auc_score(y_true, y_pred))

        memory.update_state(batch.src, batch.dst, batch.t, batch.msg)
        neighbor_loader.insert(batch.src, batch.dst)
    return float(torch.tensor(aps).mean()), float(torch.tensor(aucs).mean()), node_emb

def run(data, data_name):
    train_split = int(0.7 * len(data))
    train_data = data[:train_split]
    val_data = data[train_split:]
    # train_data, val_data, test_data = data.train_val_test_split(
    # val_ratio=0.15, test_ratio=0.15)

    # val_data = ConcatDataset([val_data, test_data])

    if (len(val_data) // 5) == 0:
        val_batch_size = len(val_data)
    else:
        val_batch_size = len(val_data) // 5


    train_loader = TemporalDataLoader(
        train_data,
        batch_size= len(train_data) // 5,
        neg_sampling_ratio=1.0,
    )
    val_loader = TemporalDataLoader(
        val_data,
        batch_size= val_batch_size,
        neg_sampling_ratio=1.0,
    )
    # test_loader = TemporalDataLoader(
    #     test_data,
    #     batch_size=200,
    #     neg_sampling_ratio=1.0,
    # )
    neighbor_loader = LastNeighborLoader(data.num_nodes, size=10, device=device)

    memory = TGNMemory(
        data.num_nodes,
        data.msg.size(-1),
        memory_dim,
        time_dim,
        message_module=IdentityMessage(data.msg.size(-1), memory_dim, time_dim),
        aggregator_module=LastAggregator(),
    ).to(device)

    gnn = GraphAttentionEmbedding(
        in_channels=memory_dim,
        out_channels=embedding_dim,
        msg_dim=data.msg.size(-1),
        time_enc=memory.time_enc,
    ).to(device)

    link_pred = LinkPredictor(in_channels=embedding_dim).to(device)

    optimizer = torch.optim.Adam(
        set(memory.parameters()) | set(gnn.parameters())
        | set(link_pred.parameters()), lr=0.0001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Helper vector to map global node indices to local ones.
    assoc = torch.empty(data.num_nodes, dtype=torch.long, device=device)

    for epoch in range(1, num_epochs + 1):
        best_val_ap = 0.0
        best_node_emb = None
        best_graph_emb = None

        print("Epoch: {}".format(epoch))
        print("Training ...\n")
        loss, train_node_emb = train(train_data, gnn, memory, link_pred, train_loader, neighbor_loader, assoc, criterion, optimizer)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}')
        print("Validating ...\n")
        val_ap, val_auc, val_node_emb = test(gnn, memory, link_pred, assoc, neighbor_loader, val_loader)
        print(f'Val AP: {val_ap:.4f}, Val AUC: {val_auc:.4f}')
        # print("Testing ...\n")
        # test_ap, test_auc, test_node_emb = test(gnn, memory, link_pred, assoc, neighbor_loader, test_loader)
        # print(f'Test AP: {test_ap:.4f}, Test AUC: {test_auc:.4f}')

        if val_ap > best_val_ap:
            best_val_ap = val_ap
            best_node_emb = torch.cat([train_node_emb, val_node_emb], dim = 0)
            best_graph_emb = torch.sum(best_node_emb, dim = 0).detach().cpu()

        # torch.save(best_graph_emb, "{}_graphs_emb.pt".format(data_name))

        print("-------------------\n")
    return best_graph_emb

parser = argparse.ArgumentParser()
parser.add_argument('--graphs_dataset', type = str)

args = parser.parse_args()

if args.graphs_dataset.startswith("dblp"):
    num_graphs = 755
if args.graphs_dataset.startswith("facebook"):
    num_graphs = 995
if args.graphs_dataset.startswith("tumblr"):
    num_graphs = 373
if args.graphs_dataset.startswith("infectious"):
    num_graphs = 200
if args.graphs_dataset.startswith("highschool"):
    num_graphs = 180

if __name__ == "__main__":
    
    for data_prefix in ["facebook_ct1"]:
    # for data_prefix in ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1"]:
        for snapshot_id in range(4):
            args.graphs_dataset = data_prefix + "_" + str(snapshot_id)

            if args.graphs_dataset.startswith("dblp"):
                num_graphs = 755
            if args.graphs_dataset.startswith("facebook"):
                num_graphs = 995
            if args.graphs_dataset.startswith("tumblr"):
                num_graphs = 373
            if args.graphs_dataset.startswith("infectious"):
                num_graphs = 200
            if args.graphs_dataset.startswith("highschool"):
                num_graphs = 180

            data_name = args.graphs_dataset
            all_graphs_emb = torch.Tensor([])
            device = "cpu"
            start_time = time.time()
            for i in range(num_graphs):
                torch.cuda.empty_cache()
                data = torch.load("./datasets/{}/{}/data.pt".format(data_name, i))
                msg = torch.zeros((data.t.shape)).view(data.t.shape[0], -1)
                data = TemporalData(src = data.src, dst = data.dst, t = data.t, msg = msg)
                data = data.to(device)
                graph_emb = run(data, data_name)
                all_graphs_emb = torch.cat((all_graphs_emb, graph_emb), dim = 0)

            torch.save(all_graphs_emb, "{}_graphs_emb.pt".format(data_name))

            total_time = time.time() - start_time

            print(f'{args.graphs_dataset}: {total_time:.2f}')
