from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToGlobalsAggregator
from wmpgnn.blocks.aggregators import NodesToGlobalsAggregator
import torch


class GlobalBlock(AbstractModule):
    """
    Module that updates graph‐level (global) features by aggregating edge and node features
    and passing the concatenated result through a learnable global model.

    This block can optionally include:
      - Edge‐based aggregation (via `EdgesToGlobalsAggregator`).
      - Node‐based aggregation (via `NodesToGlobalsAggregator`).
      - Existing global features.

    Aggregation can be weighted by edge and node probability scores if `weighted_mp=True`.

    Attributes:
        _use_edges (bool): Whether to include edge‐based aggregation.
        _use_nodes (bool): Whether to include node‐based aggregation.
        _use_globals (bool): Whether to include existing global features.
        _global_model (nn.Module): The module returned by `global_model_fn()`,
            called on the concatenated aggregate vector.
        _edges_aggregator (EdgesToGlobalsAggregator): Aggregates per‐edge features
            into a graph‐level tensor (if `_use_edges`).
        _nodes_aggregator (NodesToGlobalsAggregator): Aggregates per‐node features
            into a graph‐level tensor (if `_use_nodes`).
    """

    def __init__(self, global_model_fn, use_edges=True, use_nodes=True, use_globals=True, weighted_mp=False):
        """
        Initialize the GlobalBlock.

        Args:
            global_model_fn (callable): Zero‐argument function that returns an `nn.Module`
                (e.g., an MLP) which accepts a single Tensor of shape [n_graphs, dim_global_in]

            use_edges (bool): Include edge‐aggregated features in the input to the global model.
            use_nodes (bool): Include node‐aggregated features in the input to the global model.
            use_globals (bool): Include existing global features in the input to the global model.
            weighted_mp (bool): If True, pass per‐edge and/or per‐node weights to the aggregators.
        """

        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgesToGlobalsAggregator(weighted=weighted_mp)

            if self._use_nodes:
                self._nodes_aggregator = NodesToGlobalsAggregator(weighted=weighted_mp)



    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        """
        Compute updated edge features and return a new graph with updated edges.

        The forward pass will:
          1. Collect the requested feature tensors for each edge.
          2. Concatenate them along the last dimension.
          3. Call `self._edge_model(concatenated_inputs, graph.edgepos)`.
          4. Update `graph.edges` with the result via `graph.update()`.

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
            The same `graph` object, with its `edges` attribute replaced by the
            Tensor returned by the edge model: shape [E, D_e_out].
        """

        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph, edge_weights))


        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph, node_weights))

        if self._use_globals:
            globals_to_collect.append(graph.graph_globals)


        collected_globals = torch.cat(globals_to_collect, axis=-1)




        updated_globals = self._global_model(collected_globals)

        graph.update({'graph_globals':  updated_globals})

        return graph