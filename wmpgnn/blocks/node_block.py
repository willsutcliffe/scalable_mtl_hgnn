from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToNodesAggregator
from wmpgnn.blocks.aggregators import globals_to_nodes
import torch


class NodeBlock(AbstractModule):
    def __init__(self, node_model_fn, use_sender_edges=True,
                 use_receiver_edges=False, use_globals=True, use_nodes=True):
        # for undirected graph, set use_received_edges=False
        super(NodeBlock, self).__init__()

        self._use_sender_edges = use_sender_edges
        self._use_receiver_edges = use_receiver_edges
        self._use_globals = use_globals
        self._use_nodes = use_nodes

        with self._enter_variable_scope():
            self._node_model = node_model_fn()

            if self._use_receiver_edges:
                #                 if received_edges_reducer is None:
                #                     raise ValueError(
                #                       "If `use_received_edges==True`, `received_edges_reducer`should not be None.")
                self._received_edges_aggregator = EdgesToNodesAggregator()

            if self._use_sender_edges:
                #                 if sent_edges_reducer is None:
                #                     raise ValueError(
                #                         "If `use_sent_edges==True`, `sent_edges_reducer` should not be None.")
                self._sent_edges_aggregator = EdgesToNodesAggregator(use_sent_edges=True)

    def forward(self, graph, edge_weights):
        nodes_to_collect = []

        if self._use_sender_edges:
            nodes_to_collect.append(self._sent_edges_aggregator(graph, edge_weights))
        #             print(self._sent_edges_aggregator(graph))

        if self._use_receiver_edges:  # should be set as False if undirected graph??
            nodes_to_collect.append(self._received_edges_aggregator(graph, edge_weights))

        if self._use_nodes:
            nodes_to_collect.append(graph.nodes)

        if self._use_globals:
            nodes_to_collect.append(globals_to_nodes(graph))

        #         print("collected nodes ", nodes_to_collect)
        collected_nodes = torch.cat(nodes_to_collect, axis=-1)
        updated_nodes = self._node_model(collected_nodes)

        # replace the original grapg
        graph.update({'nodes': updated_nodes})
        return graph

