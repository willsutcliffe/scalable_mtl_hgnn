
from wmpgnn.blocks.abstract_module import AbstractModule
import torch
from wmpgnn.blocks.aggregators import receiver_nodes_to_edges
from wmpgnn.blocks.aggregators import sender_nodes_to_edges
from wmpgnn.blocks.aggregators import globals_to_edges


class EdgeBlock(AbstractModule):
    def __init__(self, edge_model_fn,
                 use_edges=True,
                 use_receiver_nodes=True,
                 use_sender_nodes=True,
                 use_globals=True):

        super(EdgeBlock, self).__init__()

        self._use_edges = use_edges
        self._use_receiver_nodes = use_receiver_nodes
        self._use_sender_nodes = use_sender_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._edge_model = edge_model_fn()

    def forward(self, graph):
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

