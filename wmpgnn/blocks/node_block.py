from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToNodesAggregator
from wmpgnn.blocks.aggregators import globals_to_nodes
import torch
from torch_scatter import scatter_add
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_scatter import scatter_std


class NodeBlock(AbstractModule):
    def __init__(self, node_model_fn, use_sender_edges=True,
                 use_receiver_edges=False, use_globals=True, use_nodes=True, weighted_mp=False):
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
        updated_nodes = self._node_model(collected_nodes)

        graph.update({'nodes': updated_nodes})
        return graph

