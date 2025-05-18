from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToNodesAggregator
from wmpgnn.blocks.aggregators import globals_to_nodes
import torch


class NodeBlock(AbstractModule):

    """
    Module that updates node features by aggregating incoming and/or outgoing
    edge messages, existing node features, and global features, then applying
    a learnable node model.

    This block supports:
      - Aggregation of sender‐edge contributions (edges sent by the node).
      - Aggregation of receiver‐edge contributions (edges received by the node).
      - Inclusion of the node's current features.
      - Inclusion of graph‐level global features.

    Message‐passing weights can be applied if `weighted_mp=True`.

    Attributes:
        _use_sender_edges (bool): Include messages from sender edges.
        _use_receiver_edges (bool): Include messages from receiver edges.
        _use_nodes (bool): Include existing node features.
        _use_globals (bool): Include graph‐level global features.
        _node_model (nn.Module): The module returned by `node_model_fn()`, called
            with `(collected_node_inputs, graph.batch)`.
        _sent_edges_aggregator (EdgesToNodesAggregator): Aggregates messages
            from outgoing edges (if `_use_sender_edges`).
        _received_edges_aggregator (EdgesToNodesAggregator): Aggregates messages
            from incoming edges (if `_use_receiver_edges`).
    """

    def __init__(self, node_model_fn, use_sender_edges=True,
                 use_receiver_edges=False, use_globals=True, use_nodes=True, weighted_mp=False):
        """
        Initialize the NodeBlock.

        Args:
            node_model_fn (callable): Zero‐argument function that returns an `nn.Module`
                (e.g., an MLP) which accepts:
                  - `node_inputs`: Tensor [N, D_n_in], concatenated inputs per node.
                Returns: Tensor [N, D_n_out] of updated node features.
            use_sender_edges (bool): Include aggregation over edges where the node is the sender.
            use_receiver_edges (bool): Include aggregation over edges where the node is the receiver.
            use_globals (bool): Include per‐graph global features via `globals_to_nodes(graph)`.
            use_nodes (bool): Include the node's existing feature vector `graph.nodes`.
            weighted_mp (bool): If True, pass edge weights to the aggregators for weighted message passing.
        """
        super(NodeBlock, self).__init__()

        self._use_sender_edges = use_sender_edges
        self._use_receiver_edges = use_receiver_edges
        self._use_globals = use_globals
        self._use_nodes = use_nodes

        with self._enter_variable_scope():
            self._node_model = node_model_fn()

            if self._use_receiver_edges:
                self._received_edges_aggregator = EdgesToNodesAggregator(weighted=weighted_mp)

            if self._use_sender_edges:
                self._sent_edges_aggregator = EdgesToNodesAggregator(use_sent_edges=True, weighted=weighted_mp)

    def forward(self, graph, edge_weights):
        """
        Compute updated node features and return the graph with updated nodes.

        The forward pass will:
          1. Aggregate outgoing edge messages if enabled:
             `sent_msgs = self._sent_edges_aggregator(graph, edge_weights)`
          2. Aggregate incoming edge messages if enabled:
             `recv_msgs = self._received_edges_aggregator(graph, edge_weights)`
          3. Optionally include the node's existing features: `graph.nodes`.
          4. Optionally include graph‐level globals: `globals_to_nodes(graph)`.
          5. Concatenate all selected inputs along the feature dimension.
          6. Call `self._node_model(collected_inputs, graph.batch)`.
          7. Update `graph.nodes` via `graph.update()`.

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
            The same `graph` object with its `nodes` attribute replaced by the
            Tensor returned by the node model: shape [N, D_out].
        """

        nodes_to_collect = []

        if self._use_sender_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph, edge_weights))


        if self._use_receiver_edges:
            nodes_to_collect.append(self._received_edges_aggregator(graph, edge_weights))


        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            nodes_to_collect.append(globals_to_nodes(graph))


        collected_nodes = torch.cat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes, graph.batch)

        graph.update({'nodes': updated_nodes})
        return graph

