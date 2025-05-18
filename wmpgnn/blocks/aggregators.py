from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_scatter import scatter_std

def globals_to_nodes(graph):
    """
    Broadcast graph‑level features to each node in a batched PyTorch Geometric graph.

    Args:
    graph (torch_geometric.data.Batch): A batched graph object with per-graph
        global features.

    Returns:
        torch.Tensor: A Tensor of shape [N, D_g] where each row contains the global
        feature vector of the graph that the corresponding node belongs to.
    """
    return graph.graph_globals[graph.batch]

def receiver_nodes_to_edges(graph):
    """
    Gather receiver‐node features for each edge in a batched PyTorch Geometric graph.

    Args:
        graph (torch_geometric.data.Batch or Data): A graph object whose edges are
            described by `graph.receivers`.

    Returns:
        torch.Tensor: A Tensor of shape [E, D_n] the receiver node features
        of an each edge
    """
    return graph.nodes[graph.receivers, :]

def sender_nodes_to_edges(graph):
    """
    Gather sender‐node features for each edge in a batched PyTorch Geometric graph.

    Args:
        graph (torch_geometric.data.Batch or Data): A graph object whose edges are
            described by `graph.receivers`.

    Returns:
        torch.Tensor: A Tensor of shape [E, D_n] the receiver node features
        of an each edge
    """
    return graph.nodes[graph.senders, :]

def globals_to_edges(graph):
    """
    Broadcast graph‑level features to each edge in a batched PyTorch Geometric graph.

    Args:
    graph (torch_geometric.data.Batch): A batched graph object with per-graph features and
    batch indices for each node and edge.

    Returns:
        torch.Tensor: A Tensor of shape [E, D_g] where each row contains the global
        feature vector of the graph that the corresponding edge belongs to.
    """
    return graph.graph_globals[graph.edgepos]

class EdgesToNodesAggregator(AbstractModule):
    def __init__(self, use_sent_edges=False, b_edges=True, weighted=True, scatter_func = scatter_add):
        super(EdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        if graph.nodes is not None and graph.nodes.size()[0] is not None:
            num_nodes = graph.nodes.size()[0]
        else:
            num_nodes = graph.num_nodes

        indices = graph.senders if self._use_sent_edges else graph.receivers
        out = graph.edges.new_zeros(num_nodes, graph.edges.shape[1])

        if not self._weighted:
            return self._scatter_func(graph.edges, indices, out=out, dim=0)
        elif self._b_edges == True:
            return self._scatter_func(graph.edges * edge_weights, indices, out=out, dim=0)
        else:
            return self._scatter_func(graph.edges * (1 - edge_weights), indices, out=out, dim=0)


class EdgesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None, weighted = True, b_edges=True, scatter_func = scatter_add):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.edges, 0)
        else:
            out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.edges.shape[1])
            if not self._weighted:
                out = self._scatter_func(graph.edges, graph.edgepos,out=out, dim=0)
            elif self._b_edges:
                out = self._scatter_func(graph.edges*edge_weights, graph.edgepos, out=out, dim=0)
            else:
                out = self._scatter_func(graph.edges * (1-edge_weights), graph.edgepos, out=out, dim=0)

        return out


class NodesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None, weighted=True, b_nodes=True, scatter_func = scatter_add):
        super(NodesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1
        self._b_nodes = b_nodes
        self._weighted = weighted
        self._scatter_func = scatter_func
    def forward(self, graph, node_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.nodes, 0)
        else:
            out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.nodes.shape[1])
            if not self._weighted:
                out = self._scatter_func(graph.nodes, graph.batch, out=out, dim=0)
            elif self._b_nodes:
                out = self._scatter_func(graph.nodes*node_weights, graph.batch, out=out, dim=0)
            else:
                out = self._scatter_func(graph.nodes * (1-node_weights), graph.batch, out=out, dim=0)
        return out



