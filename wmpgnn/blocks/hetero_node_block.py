from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_aggregators import HeteroEdgesToNodesAggregator

import torch


class HeteroNodeBlock(AbstractModule):
    def __init__(self, node_types, edge_types, node_model_fn, use_sender_edges=True,
                 use_receiver_edges=False, use_globals=True, use_nodes=True, weighted_mp=False):
        # for undirected graph, set use_received_edges=False
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
            updated_nodes = self._node_models[node_type](collected_nodes)

            graph[node_type].x = updated_nodes
        return graph