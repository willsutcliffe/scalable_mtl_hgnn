from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_aggregators import HeteroEdgesToGlobalsAggregator
from wmpgnn.blocks.hetero_aggregators import HeteroNodesToGlobalsAggregator
import torch

class HeteroGlobalBlock(AbstractModule):
    """
    Module that updates graph‐level (global) features for a heterogeneous graph by
    aggregating contributions from each relation and node type.

    For each edge type and node type, this block:
      1. Uses a heterogeneous aggregator to pool edge or node features into per‐graph summaries.
      2. Optionally includes the existing global features.
      3. Concatenates all collected tensors.
      4. Feeds the result through a shared learnable global model.
      5. Writes the updated globals back into `graph['globals'].x`.

    Attributes:
        _use_edges (bool): Whether to include aggregated edge contributions.
        _use_nodes (bool): Whether to include aggregated node contributions.
        _use_globals (bool): Whether to include existing global features.
        _node_types (Iterable[str]): List of node set keys in `graph`.
        _edge_types (Iterable[tuple]): List of edge relation keys `(src, dst, rel)`.
        _global_model (nn.Module): The module returned by `global_model_fn()`,
            applied to the concatenated aggregates.
        _edges_aggregator (HeteroEdgesToGlobalsAggregator): Aggregates per‐relation
            edge features into graph summaries (if `_use_edges`).
        _nodes_aggregator (HeteroNodesToGlobalsAggregator): Aggregates per‐node‐type
            features into graph summaries (if `_use_nodes`).
    """
    def __init__(self, node_types, edge_types, global_model_fn, use_edges=True,
                 use_nodes=True, use_globals=True, weighted_mp=False):
        """
        Initialize the HeteroGlobalBlock.

        Args:
            node_types (Iterable[str]): Keys identifying node sets in `graph`.
            edge_types (Iterable[tuple]): Keys identifying edge relations `(src_type, dst_type, rel_key)`.
            global_model_fn (callable): Zero‐arg function returning an `nn.Module` which accepts:
                - `global_inputs`: Tensor [G, D_in], concatenated features per graph.
                Returns: Tensor [G, D_out].
            use_edges (bool): Include edge‐type aggregates via `HeteroEdgesToGlobalsAggregator`.
            use_nodes (bool): Include node‐type aggregates via `HeteroNodesToGlobalsAggregator`.
            use_globals (bool): Include the existing `graph['globals'].x` features.
            weighted_mp (bool): If True, pass per‐edge/node weights into the aggregators.
        """
        super(HeteroGlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        self._node_types = node_types
        self._edge_types = edge_types

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = HeteroEdgesToGlobalsAggregator(weighted=weighted_mp)

            if self._use_nodes:
                self._nodes_aggregator = HeteroNodesToGlobalsAggregator(weighted=weighted_mp)



    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        """
        Compute updated global features and return the heterogeneous graph with updated globals.

        Steps:
          1. For each edge type in `_edge_types`, aggregate its edges:
             `edges_agg = self._edges_aggregator(graph, edge_type, edge_weights[edge_type])`
          2. For each node type in `_node_types`, aggregate its nodes:
             `nodes_agg = self._nodes_aggregator(graph, node_type, node_weights[node_type])`
          3. Optionally append the existing global features: `graph['globals'].x`.
          4. Concatenate all collected tensors along the feature dimension.
          5. Call `self._global_model(concatenated, **(global_model_kwargs or {}))`.
          6. Assign the result Tensor [G, D_out] to `graph['globals'].x`.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.
            edge_weights (dict): Maps each edge_type to a Tensor [E_t] of weights.
            node_weights (dict): Maps each node_type to a Tensor [N_t] of weights.
            global_model_kwargs (dict, optional): Extra keyword args for the global model.

        Returns:
            The same `graph` object with its `graph['globals'].x` replaced by the updated globals.
        """

        globals_to_collect = []

        if self._use_edges:
            for edge_type in self._edge_types:
                globals_to_collect.append(self._edges_aggregator(graph, edge_type, edge_weights[edge_type]))

        if self._use_nodes:
            for node_type in self._node_types:
                globals_to_collect.append(self._nodes_aggregator(graph, node_type, node_weights[node_type]))

        if self._use_globals:
            globals_to_collect.append(graph['globals'].x)


        collected_globals = torch.cat(globals_to_collect, axis=-1)




        updated_globals = self._global_model(collected_globals)

        graph['globals'].x = updated_globals


        return graph