import torch
import torch.nn as nn
from graphgallery.nn.layers.pytorch import Sequential, activations, InnerProductDecoder
from torch_geometric.nn import RGCNConv


class RGCN(nn.Module):
    def __init__(self,
                 in_features, *,
                 out_features=16,
                 num_relations,
                 num_bases=30,
                 num_blocks=5, 
                 hids=[32],
                 acts=['relu'],
                 dropout=0.5,
                 bias=False):
        super().__init__()

        conv = []
        for hid, act in zip(hids, acts):
            conv.append(RGCNConv(in_features,
                                   hid,
                                   num_relations,
                                   num_bases=num_bases,
                                   num_blocks=num_blocks,
                                   cached=True,
                                   bias=bias,
                                   normalize=True))
            conv.append(activations.get(act))
            conv.append(nn.Dropout(dropout))
            in_features = hid

        conv.append(RGCNConv(in_features,
                               out_features,
                               cached=True,
                               bias=bias,
                               normalize=True))

        self.conv = Sequential(*conv)
        self.decoder = DistMultDecoder(num_relations, hid)

    def forward(self, x, edge_index, edge_type, edge_weight=None):
        z = self.conv(x, edge_index, edge_type, edge_weight)
        return z

    def cache_clear(self): 
        for conv in self.conv:
            if isinstance(conv, RGCNConv):
                conv._cached_edge_index = None
                conv._cached_adj_t = None
