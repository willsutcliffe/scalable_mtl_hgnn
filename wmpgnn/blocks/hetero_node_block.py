from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_aggregators import HeteroEdgesToNodesAggregator

import torch


class HeteroNodeBlock(AbstractModule):
    """
     Module that updates node features separately for each node type in a heterogeneous graph.

     For each node type, this block:
       1. Identifies all incident edge types (incoming, outgoing, or self‑loops).
       2. Aggregates edge messages using weighted message passing per relation.
       3. Optionally includes existing node features and global graph features.
       4. Concatenates all collected inputs and passes them through a learnable
          node model specific to that node type.
       5. Writes the updated features back into `graph[node_type].x`.

     Attributes:
         _node_types (Iterable[str]): List of node‑type keys to update.
         _edge_types (Iterable[tuple]): List of edge‑type keys (src, dst, rel).
         _node_to_edge_types (dict): Maps each node type to its incident edge types.
         _use_sender_edges (bool): Whether to include outgoing edge messages.
         _use_receiver_edges (bool): Whether to include incoming edge messages.
         _use_nodes (bool): Whether to include existing node features.
         _use_globals (bool): Whether to include per‑graph global features.
         _node_models (dict): Maps each node type to its own `nn.Module`.
         _node_models_model_dict (ModuleDict): Container for node models for parameter registration.
         _sent_edges_aggregator (HeteroEdgesToNodesAggregator): Aggregates outgoing messages.
         _received_edges_aggregator (HeteroEdgesToNodesAggregator): Aggregates incoming messages.
     """
    def __init__(self, node_types, edge_types, node_model_fn, use_sender_edges=True,
                 use_receiver_edges=False, use_globals=True, use_nodes=True, weighted_mp=False):
        """
        Initialize the HeteroNodeBlock.

        Args:
            node_types (Iterable[str]): Keys identifying node sets in `graph`.
            edge_types (Iterable[tuple]): Tuples `(src_type, dst_type, rel_key)` identifying relations.
            node_model_fn (callable): Zero‑arg function returning an `nn.Module` which accepts:
                - `node_inputs`: Tensor [N_t, D_in] for nodes of type t.
                - `node_graph_idx`: LongTensor [N_t], batch assignment per node.
              Returns: Tensor [N_t, D_out].
            use_sender_edges (bool): Include messages aggregated from outgoing edges.
            use_receiver_edges (bool): Include messages aggregated from incoming edges.
            use_globals (bool): Include global graph features (`graph['globals'].x`).
            use_nodes (bool): Include the node’s current feature vector.
            weighted_mp (bool): If True, pass per‑edge weights to aggregators.
        """
        super(HeteroNodeBlock, self).__init__()

        self._use_sender_edges = use_sender_edges
        self._use_receiver_edges = use_receiver_edges
        self._use_globals = use_globals
        self._use_nodes = use_nodes
        self._node_types = node_types
        # self._node_types.remove('globals')
        self._edge_types = edge_types
        self._node_to_edge_types = {}
        for node_type in self._node_types:
            self._node_to_edge_types[node_type] = []
            for edge_type in self._edge_types:
                if edge_type[0] == node_type or edge_type[2] == node_type:
                    self._node_to_edge_types[node_type].append(edge_type)

        self._node_models = {}
        with self._enter_variable_scope():

            for node_type in self._node_types:
                self._node_models[node_type] = node_model_fn()

            if self._use_receiver_edges:
                self._received_edges_aggregator = HeteroEdgesToNodesAggregator(weighted=weighted_mp)
            self._received_edges_aggregator = HeteroEdgesToNodesAggregator(weighted=weighted_mp)
            if self._use_sender_edges:
                self._sent_edges_aggregator = HeteroEdgesToNodesAggregator(use_sent_edges=True, weighted=weighted_mp)
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_models.items()})

    def forward(self, graph, edge_weights):
        """
        Compute updated node features and return the graph with updated node sets.

        For each node type:
          1. For each incident edge type:
             - If sender==node_type: aggregate via `_sent_edges_aggregator`.
             - If receiver==node_type: aggregate via `_received_edges_aggregator`.
          2. Optionally append the existing node features: `graph[node_type].x`.
          3. Optionally append global graph features aligned by `batch`:
             `graph['globals'].x[graph[node_type].batch]`.
          4. Concatenate all inputs along the feature dimension.
          5. Call the node model for that type with `(inputs, batch)`.
          6. Assign the output Tensor [N_t, D_out] to `graph[node_type].x`.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.

        Returns:
            The same `graph` object with each `graph[node_type].x` updated.
        """
        for node_type in self._node_types:
            nodes_to_collect = []
            for edge_type in self._node_to_edge_types[node_type]:
                edge_weight = edge_weights[edge_type]
                if edge_type[0] == node_type and edge_type[2] == node_type:
                    if self._use_sender_edges:
                        nodes_to_collect.append(self._sent_edges_aggregator(graph, edge_type, edge_weight))

                    if self._use_receiver_edges:
                        nodes_to_collect.append(self._received_edges_aggregator(graph, edge_type, edge_weight))
                elif edge_type[0] == node_type:
                    nodes_to_collect.append(self._sent_edges_aggregator(graph, edge_type, edge_weight))
                elif edge_type[2] == node_type:
                    nodes_to_collect.append(self._received_edges_aggregator(graph, edge_type, edge_weight))

            if self._use_nodes:
                nodes_to_collect.append(graph[node_type].x)

            if self._use_globals:
                nodes_to_collect.append(graph['globals'].x[graph[node_type].batch])


            collected_nodes = torch.cat(nodes_to_collect, axis=-1)
            updated_nodes = self._node_models[node_type](collected_nodes, graph[node_type].batch)

            graph[node_type].x = updated_nodes
        return graph