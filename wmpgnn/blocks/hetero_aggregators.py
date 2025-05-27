from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add


class HeteroEdgesToNodesAggregator(AbstractModule):
    """
    Aggregates edge features into node features for a specific edge type.

    Attributes
    ----------
    _use_sent_edges : bool
        If True, aggregates based on sender nodes; otherwise based on receiver nodes.
    _weighted : bool
        If True, multiplies each edge feature by its corresponding weight before summation.
    _scatter_func : callable
        Function (e.g. `torch_scatter.scatter_add`) used to scatter/sum features.
    """
    def __init__(self, use_sent_edges=False, weighted=True, scatter_func=scatter_add):
        """
        Initializes the edges-to-nodes aggregator.

        Args:
            use_sent_edges : bool, optional
                If True, aggregate using edge_index[0] (senders); if False, edge_index[1] (receivers).
                Default is False.
            weighted : bool, optional
                If True, apply per-edge weights during aggregation. Default is True.
            scatter_func : callable, optional
                Scatter function to use for aggregation (must accept `src, index, out, dim`).
                Default is `scatter_add`.
        """
        super(HeteroEdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges
        self._weighted = weighted

        self._scatter_func = scatter_func

    def forward(self, graph, edge_type, weight):
        """
        Perform the aggregation of edge features into node features.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.
            edge_type : tuple
                A triplet (src_node_type, relation, dst_node_type) identifying which edges
                to aggregate.
            weight : Tensor of shape [E, 1] or [E, D_e], optional
                Per-edge weights to multiply with `graph[edge_type].edges`. Required if `weighted=True`.

        Returns:
            Tensor
                A tensor of shape [N, D_e] where N is the number of target nodes and
                D_e is the edge-feature dimensionality, containing the summed (and weighted)
                edge features per node.
        """
        indices = graph[edge_type].edge_index[0] if self._use_sent_edges else graph[edge_type].edge_index[1]
        num_nodes = graph[edge_type[0]].x.shape[0] if self._use_sent_edges else graph[edge_type[2]].x.shape[0]

        out = graph[edge_type].edges.new_zeros(num_nodes, graph[edge_type].edges.shape[1])
        if self._weighted:
            output = scatter_add(graph[edge_type].edges * weight, indices, out=out, dim=0)
        else:
            output = scatter_add(graph[edge_type].edges, indices, out=out, dim=0)
        return output

class HeteroEdgesToGlobalsAggregator(AbstractModule):
    """
    Aggregates edge features into per-graph global features for a specific edge type.

    Attributes:
        _weighted : bool
            If True, multiplies edge features by weights before summation.
        _scatter_func : callable
            Function (e.g. `torch_scatter.scatter_add`) used to scatter/sum features.
    """
    def __init__(self, num_graphs=None, scatter_func = scatter_add, weighted = True):
        """
        Initializes the edges-to-globals aggregator.

        Args:
            num_graphs : int or None, optional
                Number of graphs in the batch (unused if None; inferred from `graph['globals'].x`).
            scatter_func : callable, optional
                Scatter function to use for aggregation. Default is `scatter_add`.
            weighted : bool, optional
                If True, apply per-edge weights during aggregation. Default is True.
        """
        super(HeteroEdgesToGlobalsAggregator, self).__init__()


        self._scatter_func = scatter_func
        self._weighted = weighted

    def forward(self, graph, edge_type, weights):
        """
        Perform the aggregation of edge features into global graph features.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.
            edge_type : tuple
                Triplet (src_node_type, relation, dst_node_type) identifying which edges
                to aggregate.
            weights : Tensor of shape [E, 1] or [E, D_e], optional
                Per-edge weights to multiply with features, if `weighted=True`.

        Returns:
            Tensor
                A tensor of shape [G, D_e] where G is the number of graphs in the batch
                and D_e is the edge-feature dimensionality, containing summed edge features
                per graph.
        """
        out = graph[edge_type].edges.new_zeros(graph['globals'].x.shape[0], graph[edge_type].edges.shape[1])
        if self._weighted:
            output = self._scatter_func(graph[edge_type].edges*weights, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0] ] ,out=out, dim=0)
        else:
            output = self._scatter_func(graph[edge_type].edges, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0] ] ,out=out, dim=0)
        return output


class HeteroNodesToGlobalsAggregator(AbstractModule):
    """
    Aggregates node features into per-graph global features for a specific node type.

    Attributes:
        _weighted : bool
            If True, multiplies node features by weights before summation.
        _scatter_func : callable
            Function (e.g. `torch_scatter.scatter_add`) used to scatter/sum features.
    """
    def __init__(self, num_graphs=None, scatter_func = scatter_add, weighted = True):
        """
        Initializes the nodes-to-globals aggregator.

        Args:
            num_graphs : int or None, optional
                Number of graphs in the batch (unused if None; inferred from `graph['globals'].x`).
            scatter_func : callable, optional
                Scatter function to use for aggregation. Default is `scatter_add`.
            weighted : bool, optional
                If True, apply per-node weights during aggregation. Default is True.
        """
        super(HeteroNodesToGlobalsAggregator, self).__init__()
        self._weighted = weighted
        self._scatter_func = scatter_func
    def forward(self, graph, node_type, weights):
        """
        Perform the aggregation of node features into global graph features.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.
            node_type : str
                The node-type key identifying which nodes to aggregate (e.g., "tracks").
            weights : Tensor of shape [N, 1] or [N, D_n], optional
                Per-node weights to multiply with features, if `weighted=True`.

        Returns:
            Tensor
                A tensor of shape [G, D_n] where G is the number of graphs in the batch
                and D_n is the node-feature dimensionality, containing summed node features
                per graph.
        """
        out = graph[node_type].x.new_zeros(graph['globals'].x.shape[0], graph[node_type].x.shape[1])
        if self._weighted:
            output = self._scatter_func(graph[node_type].x * weights, graph[node_type].batch,out=out, dim=0)
        else:
            output = self._scatter_func(graph[node_type].x, graph[node_type].batch,out=out, dim=0)
        return output