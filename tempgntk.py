import torch
import math
import scipy
import numpy as np

class TemporalGNTK(object):
    def __next(self, S, diag_1, diag_2):
        S /= (diag_1[:, None] * diag_2[None, :])
        S = torch.clamp(S, -1, 1)
        
        dS = (math.pi - torch.arccos(S)) / (math.pi)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / (math.pi)
        S *= (diag_1[:, None] * diag_2[None, :])
        
        return S, dS

    def __next_wo_diag(self, S):
        diag = torch.sqrt(S.diag())
        
        S /= (diag[:, None] * diag[None, :])
        S = torch.clamp(S, -1, 1)

        dS = (math.pi - torch.arccos(S)) / (math.pi)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / (math.pi)
        S *= (diag[:, None] * diag[None, :])

        return S, dS

    def normalize_length(self, X):
        r"""
            Args:
                X (torch.tensor): shape [N, K, d]

            Returns:
                X (torch.tensor): vector length is normalized to 1 in the last dimension
        """
        # X: [N, K, d] -> length [N, K, 1]
        length = torch.sqrt(torch.sum(X ** 2, dim = -1)).unsqueeze(-1)
        length += (length == 0)
        # scale vector to length of 1 -> [N, K, d]
        return X / length

    def temporal_gntk(self, X_1, X_2, args):
        r"""
            Args: 
                X_1 (torch.tensor): shape [N, K, d]
                X_2 (torch.tensor): shape [N', K', d]
        """
        X_1 = self.normalize_length(X_1)
        X_2 = self.normalize_length(X_2)

        K_1, K_2 = X_1.shape[1], X_2.shape[1]
        N_1, N_2 = X_1.shape[0], X_2.shape[0]

        # [N, K, d] [N', K', d] -> [N, N', K, K']
        sigma_0 = torch.einsum("abd, mnd -> ambn", X_1, X_2)
        ntk_0 = torch.clone(sigma_0)

        dot_sigma_1, sigma_1 = self.__next(sigma_0)
        ntk_1 = ntk_0 * dot_sigma_1 + sigma_1

        # [N, N', K, K']
        link_ntk = ntk_1
        # [N, N']
        node_ntk = torch.einsum("mnpq -> mn", link_ntk) / (K_1 * K_2) 
        # scalar
        graph_ntk = torch.sum(node_ntk) / (N_1 * N_2)

        return link_ntk, node_ntk, graph_ntk
    
    def temporal_gntk_2(self, X_1, X_2, args):
        r"""
            Args: 
                X_1 (torch.tensor): shape [N, K, d]
                X_2 (torch.tensor): shape [N', K', d]
        """
        X_1 = self.normalize_length(X_1)
        X_2 = self.normalize_length(X_2)

        K_1, K_2 = X_1.shape[1], X_2.shape[1]
        N_1, N_2 = X_1.shape[0], X_2.shape[0]

        # [N, K, d] -> [N, 1, d] -> [N, d]
        X_1_agg = torch.sum(X_1, dim = 1).squeeze() / K_1
        X_2_agg = torch.sum(X_2, dim = 1).squeeze() / K_2

        # [N, N']
        sigma_0 = torch.mm(X_1_agg, X_2_agg.T) / args.time_dim
        ntk_0 = torch.clone(sigma_0)

        dot_sigma_1, sigma_1 = self.__next(sigma_0)
        ntk_1 = ntk_0 * dot_sigma_1 + sigma_1

        graph_ntk = torch.sum(ntk_1) / (N_1 * N_2)

        return graph_ntk
    
    def temporal_gntk_3(self, X_1, X_2, args):
        X_1 = self.normalize_length(X_1)
        X_2 = self.normalize_length(X_2)

        K_1, K_2 = X_1.shape[1], X_2.shape[1]
        N_1, N_2 = X_1.shape[0], X_2.shape[0]

        # [N, K, d] -> [N, 1, d] -> [N, d]
        X_1_agg = torch.sum(X_1, dim = 1).squeeze() / K_1
        X_2_agg = torch.sum(X_2, dim = 1).squeeze() / K_2

        # [N, N']
        sigma = torch.mm(X_1_agg, X_2_agg.T) / (args.time_dim)
        ntk = torch.clone(sigma)

        for _ in range(args.num_mlp_layers):
            dot_sigma, sigma = self.__next(sigma)
            ntk = ntk * dot_sigma + sigma

        node_ntk = ntk
        graph_ntk = torch.sum(node_ntk) / (N_1 * N_2)

        return graph_ntk, node_ntk
    
    def temporal_gntk_4(self, node_emb_1, node_emb_2, args):
        N_1, N_2 = node_emb_1.shape[0], node_emb_2.shape[0]
        
        # [N, d] [N', d] -> [N, N]
        node_emb_1 = self.normalize_length(node_emb_1)
        node_emb_2 = self.normalize_length(node_emb_2)
        
        sigma = torch.mm(node_emb_1, node_emb_2.T) 
        # / (args.time_dim)
        ntk = torch.clone(sigma)

        for _ in range(args.num_mlp_layers):
            sigma, dot_sigma = self.__next_wo_diag(sigma)
            ntk = ntk * dot_sigma + sigma

        graph_ntk = torch.sum(ntk) 
        # / (N_1 * N_2)

        return graph_ntk
    
    def __next_diag(self, S):
        diag = torch.sqrt(S.diag())
        S /= (diag[:, None] * diag[None, :])
        S = torch.clamp(S, -1, 1)

        dS = (math.pi - torch.arccos(S)) / (2 * math.pi)
        S = (S * (math.pi - torch.arccos(S)) + torch.sqrt(1 - S * S)) / (2 * math.pi)
        S *= (diag[:, None] * diag[None, :])

        return S, dS, diag

    def get_diag_list(self, node_emb, A, args, return_ntk = False):
        n = node_emb.shape[0]
        node_emb = self.normalize_length(node_emb)
        sigma = torch.mm(node_emb, node_emb.T).nan_to_num()

        # if n > 1000:
        #     sparse_A = A.to_sparse_coo()
        #     row, col = sparse_A.indices()
        #     vals = sparse_A.values()
        #     sparse_A = scipy.sparse.coo_array((vals, (row, col)), shape = (n, n))

        #     adj_block = np.nan_to_num(scipy.sparse.kron(sparse_A, sparse_A)).astype(np.float64)
        #     sigma = np.nan_to_num(adj_block.dot(sigma.view(-1, 1).numpy()))
        #     sigma = torch.from_numpy(sigma).view(n, n)

        # else:
        #     adj_block = torch.kron(A, A).nan_to_num()
        #     sigma = torch.mm(adj_block.to(torch.float), sigma.view(-1, 1)).view(n, n)
        #     sigma = sigma.nan_to_num()
        
        ntk = torch.clone(sigma).to(args.device)
        sigma = sigma.to(args.device)
        
        diag_list = []
        for _ in range(args.num_mlp_layers):
            sigma, dot_sigma, diag = self.__next_diag(sigma)
            sigma = sigma.nan_to_num()
            dot_sigma = dot_sigma.nan_to_num()
            diag = diag.nan_to_num()
            if args.skip_connection:
                ntk = ntk * (dot_sigma + 1) + sigma
            else:
                ntk = ntk * dot_sigma + sigma
            ntk = ntk.nan_to_num()
            diag_list.append(diag.detach().cpu())

        if return_ntk:
            return ntk.detach().cpu, diag_list

        return diag_list

    def gntk(self, node_emb_1, node_emb_2, A_1, A_2, diag_list_1, diag_list_2, args):
        n_1, n_2 = node_emb_1.shape[0], node_emb_2.shape[0]
        node_emb_1 = self.normalize_length(node_emb_1)
        node_emb_2 = self.normalize_length(node_emb_2)

        sigma = torch.mm(node_emb_1, node_emb_2.T).nan_to_num()

        # if n_1 > 1000 or n_2 > 1000:
        #     sparse_A_1, sparse_A_2 = A_1.to_sparse_coo(), A_2.to_sparse_coo()
            
        #     row, col = sparse_A_1.indices()
        #     vals = sparse_A_1.values()
        #     sparse_A_1 = scipy.sparse.coo_array((vals, (row, col)), shape = (n_1, n_1))

        #     row, col = sparse_A_2.indices()
        #     vals = sparse_A_2.values()
        #     sparse_A_2 = scipy.sparse.coo_array((vals, (row, col)), shape = (n_2, n_2))

        #     adj_block = np.nan_to_num(scipy.sparse.kron(sparse_A_1, sparse_A_2)).astype(np.float64)
        #     sigma = np.nan_to_num(adj_block.dot(sigma.view(-1, 1).numpy()))
        #     sigma = torch.from_numpy(sigma).view(n_1, n_2)
        # else:
        #     adj_block = torch.kron(A_1, A_2).nan_to_num()
        #     sigma = torch.mm(adj_block.to(torch.float), sigma.view(-1, 1)).view(n_1, n_2).nan_to_num()
        
        ntk = torch.clone(sigma).to(args.device)
        sigma = sigma.to(args.device)

        for i in range(args.num_mlp_layers):
            sigma, dot_sigma = self.__next(sigma, diag_list_1[i].to(args.device), diag_list_2[i].to(args.device))
            sigma = sigma.nan_to_num()
            dot_sigma = dot_sigma.nan_to_num()
            if args.skip_connection:
                ntk = ntk * (dot_sigma + 1) + sigma
            else:
                ntk = ntk * dot_sigma + sigma
            ntk = ntk.nan_to_num()

        if args.node_ntk:
            return ntk.detach().cpu()

        if args.mean_graph_pooling:
            return (torch.sum(ntk) / (n_1 * n_2)).detach().cpu()
        
        return torch.sum(ntk).detach().cpu()
