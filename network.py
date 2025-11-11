import torch.nn as nn
from torch.nn.functional import normalize
import torch
import torch.nn.functional as F
import numpy

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import matplotlib
class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )

    def forward(self, x):
        return self.decoder(x)

class CompMod(nn.Module):
    def __init__(self, feature_dim, output_dim):
            super(CompMod, self).__init__()
            self.feature_dim = feature_dim
            self.output_dim = output_dim

            self.projection_head = nn.Sequential(
                nn.Linear(feature_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Linear(output_dim, output_dim),
                nn.BatchNorm1d(output_dim)
            )

    def forward(self, h_c):

            comprehensive_embedding = self.projection_head(h_c)
            return comprehensive_embedding

class Network(nn.Module):
    def __init__(self, view, input_size, feature_dim, high_feature_dim, class_num, device):
        super(Network, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.fussion = CompMod(feature_dim*view, feature_dim)
        self.decoders = nn.ModuleList(self.decoders)


        self.view = view

    def forward(self, xs):
        hs = []
        qs = []
        xrs = []
        zs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)

            xr = self.decoders[v](z)

            zs.append(z)

            xrs.append(xr)
        zc = zs[0]
        # concatenation
        for v in range(self.view):
            if v!=0:
                zc=torch.cat((zc, zs[v]), dim=1)

        zf = self.fussion(zc)
        return xrs,zs,zf



class GraphFilter(torch.nn.Module):
    def __init__(self):
        super(GraphFilter, self).__init__()

    def forward(self, X):
        epsilon = 1e-10

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X = X.to(device)
        features = F.normalize(X, dim=1)
        A = features @ features.T

        # Generate a sparse matrix (as an alternative to scipy.sparse.csr_matrix).
        A[A < epsilon] = 0  # Set values smaller than epsilon to 0 to simulate
        indices = A.nonzero().t()
        values = A[indices[0], indices[1]]
        adj = torch.sparse_coo_tensor(indices, values, A.size(), device=device)

        diag_indices = torch.arange(A.size(0), device=device).unsqueeze(0).repeat(2, 1)
        diag_values = adj.to_dense().diagonal()
        diag_sparse = torch.sparse_coo_tensor(diag_indices, diag_values, adj.size(), device=device)

        # Subtract the diagonal elements.
        adj = adj - diag_sparse

        # Process the feature matrix.
        sm_fea_s = X
        adj_norm_s = preprocess_graph(adj, 1, norm='sym', renorm=True)

        for a in adj_norm_s:
            sm_fea_s = torch.sparse.mm(a, sm_fea_s)

        return sm_fea_s
