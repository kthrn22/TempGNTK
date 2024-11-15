"""
Link Prediction using Mixer
"""
import time
import torch
import numpy as np

import argparse
import os

from utils import set_seed, load_feat, load_graph

from link_pred_train_utils import link_pred_train
from link_pred_eval_utils import link_pred_eval
from data_process_utils import check_data_leakage

####################################################################
####################################################################
####################################################################
    
# define file name
def name_fn(args, mixer_configs):
    fn = 'results/%s_neighbors%d_edges%d_layers%d_%dhop'%(args.data, args.num_neighbors, args.max_edges, args.num_layers, args.sampled_num_hops)

    if args.ignore_node_feats:
        fn += '_no_node_feat'
    if args.ignore_edge_feats:
        fn += '_no_edge_feat'
        
    if 'module_spec' in mixer_configs:
        for spec in mixer_configs['module_spec']:
            fn += '_'
            if 'token' in spec.split('+'):
                fn += 't'
            if 'channel' in spec.split('+'):
                fn += 'c'
                
    if 'use_single_layer' in mixer_configs and mixer_configs['use_single_layer']:
        fn += '_perceptron'
    return fn

def print_model_info(model):
    print(model)
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    parameters = sum([np.prod(p.size()) for p in parameters]) / 1_000_000
    print('Trainable Parameters: %.3f million' % parameters)

def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='REDDIT')
    parser.add_argument('--device', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--max_edges', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=1e-6)

    parser.add_argument('--model', type=str, default='mlp_mixer')
    parser.add_argument('--neg_samples', type=int, default=1)
    parser.add_argument('--extra_neg_samples', type=int, default=5)
    parser.add_argument('--num_neighbors', type=int, default=10) # hyper-parameters K
    parser.add_argument('--sampled_num_hops', type=int, default=1)
    parser.add_argument('--hidden_dims', type=int, default=25)
    parser.add_argument('--num_layers', type=int, default=1)
    parser.add_argument('--regen_models', action='store_true')
    parser.add_argument('--check_data_leakage', action='store_true')
    
    parser.add_argument('--ignore_node_feats', action='store_true')
    parser.add_argument('--node_feats_as_edge_feats', action='store_true')
    parser.add_argument('--ignore_edge_feats', action='store_true')
    parser.add_argument('--use_onehot_node_feats', action='store_true')

    parser.add_argument('--use_graph_structure', action='store_true')
    parser.add_argument('--structure_time_gap', type=int, default=2000) # hyper-parameters T
    parser.add_argument('--structure_hops', type=int, default=1) 

    parser.add_argument('--use_node_cls', action='store_true')
    parser.add_argument('--use_cached_subgraph', action='store_true')

    # only for temp-gntk project
    parser.add_argument('--graphs_dataset', type = str)
    return parser.parse_args()


def load_all_data(args):

    # load graph
    g, df = load_graph(args.data)
    
    args.train_edge_end = df[df['ext_roll'].gt(0)].index[0]
    args.val_edge_end   = df[df['ext_roll'].gt(1)].index[0]
    args.num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
    args.num_edges = len(df)
    print('Train %d, Valid %d, Test %d'%(args.train_edge_end, 
                                         args.val_edge_end-args.train_edge_end,
                                         len(df)-args.val_edge_end))
    print('Num nodes %d, num edges %d'%(args.num_nodes, args.num_edges))

    # load feats 
    node_feats, edge_feats = load_feat(args.data)
    node_feat_dims = 0 if node_feats is None else node_feats.shape[1]
    edge_feat_dims = 0 if edge_feats is None else edge_feats.shape[1]

    # feature pre-processing
    if args.use_onehot_node_feats:
        print('>>> Use one-hot node features')
        node_feats = torch.eye(args.num_nodes)
        node_feat_dims = node_feats.size(1)

    if args.ignore_node_feats:
        print('>>> Ignore node features')
        node_feats = None
        node_feat_dims = 0

    if edge_feats is None or args.ignore_edge_feats: # By default edge feature exists
        print('>>> Ignore edge features')
        edge_feats = torch.zeros((args.num_edges, 1)) # all edge has same features
        edge_feat_dims = 1

    if node_feats != None and args.node_feats_as_edge_feats:
        print('>>> Use node features as part of edge features') 
        edge_feats = torch.cat([node_feats[df.src.values] + node_feats[df.dst.values], edge_feats], dim=1)
        edge_feat_dims = edge_feats.size(1)
        
    print('Node feature dim %d, edge feature dim %d'%(node_feat_dims, edge_feat_dims))
    
    # double check (if data leakage then cannot continue the code)
    if args.check_data_leakage:
        check_data_leakage(args, g, df)

    args.node_feat_dims = node_feat_dims
    args.edge_feat_dims = edge_feat_dims
    
    if node_feats is not None:
        node_feats = node_feats.to(args.device) # here we only move node feats to cuda, not edges because too many edges
    
    return node_feats, edge_feats, g, df, args


def load_model(args):

    # get model 
    
    edge_predictor_configs = {
        'dim_in_time': 25,
        'dim_in_node': args.node_feat_dims,
    }

    if args.model == 'mlp_mixer':
        from model import Mixer_per_node

        mixer_configs = {
            'per_graph_size'  : args.max_edges, 
            'time_channels'   : 25, 
            'input_channels'  : args.edge_feat_dims, 
            'hidden_channels' : args.hidden_dims, 
            'out_channels'    : 25,
            'num_layers'      : args.num_layers,
            'use_single_layer' : False
        }

    elif args.model == 'gat_mixer':
        from model_self_attention import Mixer_per_node

        mixer_configs = {
            'per_graph_size'  : args.max_edges, 
            'time_channels'   : 25, 
            'input_channels'  : args.edge_feat_dims, 
            'hidden_channels' : args.hidden_dims, 
            'out_channels'    : 25,
            'num_layers'      : args.num_layers,
            'heads'           : 2
        }
        
    else:
        NotImplementedError()

    model = Mixer_per_node(mixer_configs, edge_predictor_configs)
    for k, v in model.named_parameters():
        print(k, v.requires_grad)

    print_model_info(model)

    fn = name_fn(args, mixer_configs)
    args.model_fn = fn+'.pt'
    args.link_pred_result_fn = fn+'.json'
    print(fn)

    return model, args
        
####################################################################
####################################################################
####################################################################

args = get_args()

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
    # og's code 
    # args = get_args()

    # args.regen_models = True
    # args.use_graph_structure = True

    # print(args)
    
    # args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    # args.device = torch.device(args.device)
    # # args.device = torch.device('cpu')

    # set_seed(0)
    
    # ###################################################

    # # load feats + graph
    # node_feats, edge_feats, g, df, args = load_all_data(args)
        
    # ###################################################
    # # get model 
    # model, args = load_model(args)

    # ###################################################
    # # Link prediction
    
    # if os.path.exists(args.model_fn) == False or args.regen_models:
    #     print('Train link prediction task from scratch ...')
    #     # og's code
    #     # model = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)

    #     # mine
    #     model, graph_emb = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)
    #     graph_emb = torch.sum(graph_emb, dim = 0).detach().cpu()
        
    #     torch.save(model.state_dict(), args.model_fn)
        
    #     # mine
    #     torch.save(graph_emb, "{}_graph_emb.pt".format(args.data))
        
    #     print('Save model to ', args.model_fn)
    # else:
    #     print('Load model from ', args.model_fn)
    #     model.load_state_dict(torch.load(args.model_fn))
    #     model = model.to(args.device)
          
    ###################################################
    # Recall@K + MRR
    # og's code
    # link_pred_eval(model.to(args.device), args, g, df, node_feats, edge_feats)
    
    for data_prefix in ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1", "highschool_ct1"]:
    # for data_prefix in ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1"]:
        for snapshot_id in range(5):
            if snapshot_id == 4:
                snapshot_id = ""
            args.graphs_dataset = data_prefix + "_" + str(snapshot_id)

            print(args.graphs_dataset)

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

            all_graphs_emb = torch.Tensor([])

            start_time = time.time()
            for i in range(num_graphs):     
                torch.cuda.empty_cache()

                args = get_args()
                args.graphs_dataset = data_prefix + "_" + str(snapshot_id)
                args.data = "{}/{}".format(args.graphs_dataset, i)
                print(args.data)

                args.regen_models = True
                args.use_graph_structure = True

                # args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
                args.device = "cpu"
                args.device = torch.device(args.device)
                # args.device = torch.device('cpu')

                set_seed(0)
                
                ###################################################

                # load feats + graph
                node_feats, edge_feats, g, df, args = load_all_data(args)
                    
                ###################################################
                # get model 
                model, args = load_model(args)

                ###################################################
                # Link prediction
                
                if os.path.exists(args.model_fn) == False or args.regen_models:
                    print('Train link prediction task from scratch ...')
                    # og's code
                    # model = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)

                    # mine
                    model, graph_emb = link_pred_train(model.to(args.device), args, g, df, node_feats, edge_feats)
                    time_dim = model.time_feats_dim
                    graph_emb = torch.sum(graph_emb, dim = 0).detach().cpu()
                    graph_emb = torch.clone(graph_emb[:time_dim])

                    all_graphs_emb = torch.cat((all_graphs_emb, graph_emb), dim = 0)
                    
                    # torch.save(model.state_dict(), args.model_fn)
                    
                    # mine
                    print('Save model to ', args.model_fn)
                else:
                    print('Load model from ', args.model_fn)
                    model.load_state_dict(torch.load(args.model_fn))
                    model = model.to(args.device)

            torch.save(all_graphs_emb, "{}_graphs.pt".format(args.graphs_dataset))
            total_time = time.time() - start_time

            print(f'{args.graphs_dataset}: {total_time:.2f}')
