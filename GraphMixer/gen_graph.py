### Code from GraphMixer's official implementation: https://github.com/CongWeilin/GraphMixer

import argparse
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
# parser.add_argument('--data', type=str, help='dataset name')
parser.add_argument('--add_reverse', default=False, action='store_true')

parser.add_argument('--graphs_dataset', type = str)

args = parser.parse_args()
args.add_reverse = True # GraphMixer need it

print(args)

# og's code

# df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
# num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
# print('num_nodes: ', num_nodes)

# ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
# ext_full_indices = [[] for _ in range(num_nodes)]
# ext_full_ts = [[] for _ in range(num_nodes)]
# ext_full_eid = [[] for _ in range(num_nodes)]

# for idx, row in tqdm(df.iterrows(), total=len(df)):
#     src = int(row['src'])
#     dst = int(row['dst'])
    
#     ext_full_indices[src].append(dst)
#     ext_full_ts[src].append(row['time'])
#     ext_full_eid[src].append(idx)
    
#     if args.add_reverse:
#         ext_full_indices[dst].append(src)
#         ext_full_ts[dst].append(row['time'])
#         ext_full_eid[dst].append(idx)

# for i in tqdm(range(num_nodes)):
#     ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

# ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
# ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
# ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

# print('Sorting...')

# def tsort(i, indptr, indices, t, eid):
#     beg = indptr[i]
#     end = indptr[i + 1]
#     sidx = np.argsort(t[beg:end])
#     indices[beg:end] = indices[beg:end][sidx]
#     t[beg:end] = t[beg:end][sidx]
#     eid[beg:end] = eid[beg:end][sidx]


# for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
#     tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

# print('saving...')

# np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
#          indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)

for data_prefix in ["infectious_ct1", "dblp_ct1", "tumblr_ct1", "facebook_ct1"]:
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

            for i in range(num_graphs):
                args.data = "{}/{}".format(args.graphs_dataset, i)

                df = pd.read_csv('DATA/{}/edges.csv'.format(args.data))
                num_nodes = max(int(df['src'].max()), int(df['dst'].max())) + 1
                print('num_nodes: ', num_nodes)

                ext_full_indptr = np.zeros(num_nodes + 1, dtype=np.int32)
                ext_full_indices = [[] for _ in range(num_nodes)]
                ext_full_ts = [[] for _ in range(num_nodes)]
                ext_full_eid = [[] for _ in range(num_nodes)]

                for idx, row in tqdm(df.iterrows(), total=len(df)):
                    src = int(row['src'])
                    dst = int(row['dst'])
                    
                    ext_full_indices[src].append(dst)
                    ext_full_ts[src].append(row['time'])
                    ext_full_eid[src].append(idx)
                    
                    if args.add_reverse:
                        ext_full_indices[dst].append(src)
                        ext_full_ts[dst].append(row['time'])
                        ext_full_eid[dst].append(idx)

                for i in tqdm(range(num_nodes)):
                    ext_full_indptr[i + 1] = ext_full_indptr[i] + len(ext_full_indices[i])

                ext_full_indices = np.array(list(itertools.chain(*ext_full_indices)))
                ext_full_ts = np.array(list(itertools.chain(*ext_full_ts)))
                ext_full_eid = np.array(list(itertools.chain(*ext_full_eid)))

                print('Sorting...')

                def tsort(i, indptr, indices, t, eid):
                    beg = indptr[i]
                    end = indptr[i + 1]
                    sidx = np.argsort(t[beg:end])
                    indices[beg:end] = indices[beg:end][sidx]
                    t[beg:end] = t[beg:end][sidx]
                    eid[beg:end] = eid[beg:end][sidx]


                for i in tqdm(range(ext_full_indptr.shape[0] - 1)):
                    tsort(i, ext_full_indptr, ext_full_indices, ext_full_ts, ext_full_eid)

                print('saving...')

                np.savez('DATA/{}/ext_full.npz'.format(args.data), indptr=ext_full_indptr,
                        indices=ext_full_indices, ts=ext_full_ts, eid=ext_full_eid)
