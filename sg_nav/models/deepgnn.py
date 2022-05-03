import torch
import torch.nn.functional as F
from torch.nn import LayerNorm, Linear, ReLU
from torch_scatter import scatter
from tqdm import tqdm

from torch_geometric.nn import DeepGCNLayer, GENConv
from torch_geometric.nn import (
    global_add_pool,
    global_mean_pool,
    global_max_pool,
)


class DeeperGCN(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        edge_channels=None,
        block="res+",
        aggr="softmax",
    ):
        super().__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_channels:
            self.edge_encoder = Linear(edge_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = GENConv(
                hidden_channels,
                hidden_channels,
                aggr=aggr,
                t=1.0,
                learn_t=True,
                num_layers=2,
                norm="layer",
            )
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block=block, dropout=0.1, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.g_lin = Linear(hidden_channels, out_channels)
        self.n_lin = Linear(hidden_channels, out_channels + 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if len(x.size()) > 2:
            x = torch.squeeze(x, dim=0)
        x = self.node_encoder(x)
        if edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        if len(self.layers) > 0:
            x = self.layers[0].conv(x, edge_index, edge_attr)

            for layer in self.layers[1:]:
                x = layer(x, edge_index, edge_attr)

            x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        g_out = self.g_lin(x)
        if batch is None:
            g_out, _ = g_out.max(0)  # todo
        else:
            g_out = global_max_pool(g_out, batch)
        n_out = self.n_lin(x)
        return g_out, n_out


class DeeperMLP(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_layers,
        edge_channels=None,
        block="res+",
        aggr="softmax",
    ):
        super().__init__()

        self.node_encoder = Linear(in_channels, hidden_channels)
        if edge_channels:
            self.edge_encoder = Linear(edge_channels, hidden_channels)

        self.layers = torch.nn.ModuleList()
        for i in range(1, num_layers + 1):
            conv = Linear(hidden_channels, hidden_channels)
            norm = LayerNorm(hidden_channels, elementwise_affine=True)
            act = ReLU(inplace=True)

            layer = DeepGCNLayer(
                conv, norm, act, block=block, dropout=0.1, ckpt_grad=i % 3
            )
            self.layers.append(layer)

        self.g_lin = Linear(hidden_channels, out_channels)
        self.n_lin = Linear(hidden_channels, out_channels + 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if len(x.size()) > 2:
            x = torch.squeeze(x, dim=0)
        x = self.node_encoder(x)
        if edge_attr:
            edge_attr = self.edge_encoder(edge_attr)

        if len(self.layers) > 0:
            x = self.layers[0].conv(x)

            for layer in self.layers[1:]:
                x = layer(x)

            x = self.layers[0].act(self.layers[0].norm(x))
        x = F.dropout(x, p=0.1, training=self.training)

        g_out = self.g_lin(x)
        if batch is None:
            g_out, _ = g_out.max(0)  # todo
        else:
            g_out = global_max_pool(g_out, batch)
        n_out = self.n_lin(x)
        return g_out, n_out

