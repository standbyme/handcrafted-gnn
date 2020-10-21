import torch.nn as nn
import torch.nn.functional as F
from pygcn.layers import GraphConvolution


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, src, dst, edge_weight):
        x = F.relu(self.gc1(x, src, dst, edge_weight))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, src, dst, edge_weight)
        return F.log_softmax(x, dim=1)
