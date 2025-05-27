from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add

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
    """
    Aggregates edge features into node features using a scatter-based reduction.

    This module supports both incoming and outgoing edge aggregation, optional
    use of edge weights, and configurable reduction functions (e.g., sum, mean, max).

    Attributes:
        _use_sent_edges (bool): If True, aggregates over edges where the node is the sender.
                                If False, aggregates over edges where the node is the receiver.
        _b_edges (bool): Flag indicating how to apply the edge weights:
                         - If True: use `edge_weights` directly.
                         - If False: use `1 - edge_weights` for aggregation.
        _weighted (bool): Whether to apply weighting to edge features during aggregation.
        _scatter_func (callable): Scatter function used for reduction (e.g., `scatter_add`).

    Example:
        If `use_sent_edges=False`, `weighted=True`, `b_edges=True`, and `scatter_func=scatter_add`,
        this module will compute, for each node `i`:
            aggregated[i] = sum over edges (e: e.receiver == i) of (edge_features[e] * edge_weights[e])
    """
    def __init__(self, use_sent_edges=False, b_edges=True, weighted=True, scatter_func = scatter_add):
        """
        Initializes the EdgesToNodesAggregator.

        Args:
            use_sent_edges (bool): If True, aggregate from sent edges (node is sender).
                                   If False, aggregate from received edges (node is receiver).
            b_edges (bool): Determines whether to use `edge_weights` or `1 - edge_weights` for weighting.
            weighted (bool): Whether to apply weights to edge features during aggregation.
            scatter_func (callable): Function used to aggregate features, e.g., `scatter_add`, `scatter_mean`.
        """
        super(EdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        """
        Performs edge-to-node aggregation.

        Args:
            graph: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Tensor [N, D_e]: Aggregated node features from edge features.
        """
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
    """
    Aggregates edge features into graph‐level (global) features using a scatter‐based reduction.

    This module supports optional weighting of edges, configurable reduction
    functions (e.g., sum, mean, max), and handling of multiple graphs in a batch.

    Attributes:
        _num_graphs (int or None): Number of graphs in the batch. If `None`,
            the aggregator collapses all edges into a single global vector.
        _b_edges (bool): Flag indicating how to apply edge weights:
                         - If True: use `edge_weights` directly.
                         - If False: use `1 - edge_weights` for weighting.
        _weighted (bool): Whether to apply `edge_weights` to edge features.
        _scatter_func (callable): Scatter function used for reduction
                                  (e.g., `scatter_add`, `scatter_mean`).
    """
    def __init__(self, weighted = True, b_edges=True, scatter_func = scatter_add):
        """
        Initialize the EdgesToGlobalsAggregator.

        Args:
            num_graphs (int, optional): Number of graphs in the batch. If provided,
                each graph’s edges are aggregated separately into a [num_graphs, D_e]
                tensor. If `None`, all edges are aggregated into a single [D_e] vector.
            weighted (bool): Whether to apply per‐edge weights during aggregation.
            b_edges (bool): If True, use `edge_weights` directly; if False, use
                `(1 - edge_weights)` when weighting edge features.
            scatter_func (callable): Scatter function for aggregation, e.g., `scatter_add`,
                `scatter_mean`, or `scatter_max`.
        """
        super(EdgesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        """
        Perform edge‐to‐global aggregation.

        Args:
            graph: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Tensor of aggregated global features:
            - Tensor [G, D_e], where row _i_ aggregates edges with `edgepos == i`.
        """
        out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.edges.shape[1])
        if not self._weighted:
            out = self._scatter_func(graph.edges, graph.edgepos,out=out, dim=0)
        elif self._b_edges:
            out = self._scatter_func(graph.edges*edge_weights, graph.edgepos, out=out, dim=0)
        else:
            out = self._scatter_func(graph.edges * (1-edge_weights), graph.edgepos, out=out, dim=0)

        return out


class NodesToGlobalsAggregator(AbstractModule):
    """
    Aggregates node features into graph‐level (global) features using a scatter‐based reduction.

    This module supports optional weighting of node contributions, configurable reduction
    functions (e.g., sum, mean, max), and handling of multiple graphs in a batch.

    Attributes:
        _num_graphs (int or None): Number of graphs in the batch. If `None`, the aggregator
            collapses all nodes into a single global vector.
        _b_nodes (bool): Flag indicating how to apply node weights:
                         - If True: use `node_weights` directly.
                         - If False: use `1 - node_weights` for weighting.
        _weighted (bool): Whether to apply `node_weights` to node features.
        _scatter_func (callable): Scatter function used for reduction
                                  (e.g., `scatter_add`, `scatter_mean`).
    """
    def __init__(self, weighted=True, b_nodes=True, scatter_func = scatter_add):
        """
        Initialize the NodesToGlobalsAggregator.

        Args:
            num_graphs (int, optional): Number of graphs in the batch. If provided,
                each graph’s nodes are aggregated separately into a [num_graphs, D_n]
                tensor. If `None`, all nodes are aggregated into a single [D_n] vector.
            weighted (bool): Whether to apply per‐node weights during aggregation.
            b_nodes (bool): If True, use `node_weights` directly; if False, use
                `(1 - node_weights)` when weighting node features.
            scatter_func (callable): Scatter function for aggregation, e.g., `scatter_add`,
                `scatter_mean`, or `scatter_max`.
        """
        super(NodesToGlobalsAggregator, self).__init__()
        self._b_nodes = b_nodes
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, node_weights):
        """
        Perform node‐to‐global aggregation.

        Args:
            graph: A graph‐like object with attributes:
                - `nodes`: Tensor [N, D_n], node feature matrix.
                - `batch`: LongTensor [N], graph index (0…G-1) for each node.
                - `graph_globals`: Tensor [G, D_g], used only to infer G if `num_graphs` is None.
            node_weights (Tensor [N]): Optional weights for each node (required if `weighted=True`).

        Returns:
            Tensor of aggregated global features:
            - Tensor [G, D_n], where row _i_ aggregates nodes with `batch == i`.
        """
        out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.nodes.shape[1])
        if not self._weighted:
            out = self._scatter_func(graph.nodes, graph.batch, out=out, dim=0)
        elif self._b_nodes:
            out = self._scatter_func(graph.nodes*node_weights, graph.batch, out=out, dim=0)
        else:
            out = self._scatter_func(graph.nodes * (1-node_weights), graph.batch, out=out, dim=0)
        return out



