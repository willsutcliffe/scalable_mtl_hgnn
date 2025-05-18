
from wmpgnn.blocks.abstract_module import AbstractModule
import torch
from wmpgnn.blocks.aggregators import receiver_nodes_to_edges
from wmpgnn.blocks.aggregators import sender_nodes_to_edges
from wmpgnn.blocks.aggregators import globals_to_edges


class EdgeBlock(AbstractModule):
    """
    Module that updates edge features by applying a learnable edge model to
    a concatenation of selected edge, node, and global attributes.

    This block gathers specified inputs for each edge—existing edge features,
    features of the sender and/or receiver nodes, and graph‐level globals—
    concatenates them, and passes them through a user‐supplied `edge_model_fn`
    to compute updated edge features.

    Attributes:
        _use_edges (bool): Whether to include existing edge features.
        _use_receiver_nodes (bool): Whether to include receiver‐node features.
        _use_sender_nodes (bool): Whether to include sender‐node features.
        _use_globals (bool): Whether to include graph‐level global features.
        _edge_model (nn.Module): The module returned by `edge_model_fn()`, called
            with `(concatenated_edge_inputs)`.

    Example:
        ```python
        def make_edge_mlp():
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        block = EdgeBlock(
            edge_model_fn=make_edge_mlp,
            use_edges=True,
            use_sender_nodes=True,
            use_receiver_nodes=False,
            use_globals=True
        )
        updated_graph = block(graph)
        ```
    """

    def __init__(self, edge_model_fn,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):
        """
        Initializes the EdgeBlock.

        Args:
            edge_model_fn (callable): A zero‐argument function that returns
                an `nn.Module` (e.g. an MLP) which accepts arguments:
                - `edge_inputs`: Tensor of shape [E, D_e_in] (concatenated inputs)
            use_edges (bool): If True, include `graph.edges` in the inputs.
            use_receiver_nodes (bool): If True, include features of receiver
                nodes via `receiver_nodes_to_edges(graph)`.
            use_sender_nodes (bool): If True, include features of sender nodes
                via `sender_nodes_to_edges(graph)`.
            use_globals (bool): If True, include graph‐level globals via
                `globals_to_edges(graph)`.
        """
        super(EdgeBlock, self).__init__()

        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._edge_model = edge_model_fn()

    def forward(self, graph):
        """
         Compute updated edge features and return a new graph with updated edges.

         The forward pass will:
           1. Collect the requested feature tensors for each edge.
           2. Concatenate them along the last dimension.
           3. Call `self._edge_model(concatenated_inputs, graph.edgepos)`.
           4. Update `graph.edges` with the result via `graph.update()`.

         Here graph.edgepos is the index of the graph the edge belongs to.

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
             Tensor returned by the edge model: shape [E, E_d_out].
         """
        edges_to_collect = []

        if self._use_edges:
            edges_to_collect.append(graph.edges)

        if self._use_receiver_nodes:
            edges_to_collect.append(receiver_nodes_to_edges(graph))

        if self._use_sender_nodes:
            edges_to_collect.append(sender_nodes_to_edges(graph))

        if self._use_globals:
            edges_to_collect.append(globals_to_edges(graph))



        collected_edges = torch.cat(edges_to_collect, axis=-1)
        updated_edges = self._edge_model(collected_edges, graph.edgepos)

        graph.update({'edges': updated_edges})

        return graph

