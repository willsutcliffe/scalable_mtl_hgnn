from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from wmpgnn.blocks.abstract_module import AbstractModule


class HeteroEdgeBlock(AbstractModule):
    """
    Module that updates edge features separately for each relation type in a heterogeneous graph.

    For each edge type tuple `(src_node_type, dst_node_type, edge_type)`, this block:
      1. Gathers the chosen feature components for each edge:
         - Existing edge features.
         - Receiver node features.
         - Sender node features.
         - Global graph features.
      2. Concatenates these components.
      3. Applies a learnable model specific to that edge type.
      4. Writes the updated features back into `graph[edge_type].edges`.

    Attributes:
        _edge_types (Iterable[tuple]): List of edge‑type keys, each a 3‑tuple
            `(src_type, dst_type, edge_key)`.
        _use_edges (bool): If True, include the original edge features.
        _use_receiver_nodes (bool): If True, include features of the destination nodes.
        _use_sender_nodes (bool): If True, include features of the source nodes.
        _use_globals (bool): If True, include per‑graph global features.
        _edge_models (dict): Maps each edge_type to its own `nn.Module`.
        _edge_models_model_dict (ModuleDict): PyTorch container for the edge models,
            enabling proper registration of parameters.
    """
    def __init__(self, edge_types, edge_model_fn,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        """
        Initialize the HeteroEdgeBlock.

        Args:
            edge_types (Iterable[tuple]): List of edge‑type identifiers. Each identifier
                is a tuple `(src_node_type, dst_node_type, edge_key)` matching the keys
                used in the `graph` object (e.g., `('user', 'item', 'buys')`).
            edge_model_fn (callable): Zero‑argument function that returns an `nn.Module`
                (e.g., an MLP). Each module should accept two arguments:
                  - `edge_inputs`: Tensor [E_t, D_in], concatenated features for edges of type t.
                  - `edge_batch_idx`: LongTensor [E_t], graph indices for each edge.
                The module must return Tensor [E_t, D_out].
            use_edges (bool): Include the original edge feature tensor.
            use_receiver_nodes (bool): Include features of the destination node (`dst_node_type`).
            use_sender_nodes (bool): Include features of the source node (`src_node_type`).
            use_globals (bool): Include global graph features via `graph['globals'].x`.
        """
        super(HeteroEdgeBlock, self).__init__()
        self._edge_types = edge_types
        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals
        self._edge_models = {}
        with self._enter_variable_scope():
            for edge_type in edge_types:
                self._edge_models[edge_type] = edge_model_fn()
        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_models.items()})

    def forward(self, graph):
        """
        Update the edge features for each edge type and return the modified graph.

        For each `edge_type` in `self._edge_types`:
          1. Retrieve `graph[edge_type]`, which must have:
             - `edges`: Tensor [E_t, D_e] of edge features.
             - `edge_index`: LongTensor [2, E_t], with source indices at row 0
               and destination indices at row 1.
          2. Gather requested feature tensors:
             - `edges.edges` if `use_edges`.
             - `graph[dst_type].x[...]` if `use_receiver_nodes`.
             - `graph[src_type].x[...]` if `use_sender_nodes`.
             - `graph['globals'].x[...]` if `use_globals`.
          3. Concatenate all gathered tensors along feature dimension.
          4. Call the corresponding edge model with:
             - concatenated inputs,
             - `graph[src_type].batch[edge_index[0]]` as batch indices.
          5. Assign the returned Tensor [E_t, D_out] back to `graph[edge_type].edges`.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.

        Returns:
            The same `graph` object with each `graph[edge_type].edges` updated
            to the output of the per‑type edge model.
        """
        for edge_type in self._edge_types:
            edges_to_collect = []
            edges = graph[edge_type]

            if self._use_edges:
                edges_to_collect.append(edges.edges)
            if self._use_receiver_nodes:
                edges_to_collect.append(graph[edge_type[2]].x[edges.edge_index[1], :])
            node_0 = graph[edge_type[0]]
            if self._use_sender_nodes:
                edges_to_collect.append(node_0.x[edges.edge_index[0], :])

            if self._use_globals:
                edges_to_collect.append(graph['globals'].x[node_0.batch[edges.edge_index[0]]])

            collected_edges = torch.cat(edges_to_collect, axis=-1)
            updated_edges = self._edge_models[edge_type](collected_edges, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0] ] )
            graph[edge_type].edges = updated_edges

        return graph