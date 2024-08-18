from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToGlobalsAggregator
from wmpgnn.blocks.aggregators import NodesToGlobalsAggregator
import torch


class GlobalBlock(AbstractModule):


    def __init__(self, global_model_fn, use_edges=True, use_nodes=True, use_globals=True):
        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgesToGlobalsAggregator()
            if self._use_nodes:
                self._nodes_aggregator = NodesToGlobalsAggregator()


    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        #         if global_model_kwargs is None:
        #             global_model_kwargs = {}

        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph, edge_weights))

        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph, node_weights))

        if self._use_globals:
            globals_to_collect.append(graph.graph_globals)

        # print("collected globals ", globals_to_collect)
        # print("shape nodes globals ", self._nodes_aggregator(graph).shape )
        # print("global globals ", graph.graph_globals.shape )
        collected_globals = torch.cat(globals_to_collect, axis=-1)
        # collected_globals = torch.unsqueeze(collected_globals, 0)
        # print("collected globals shape ",collected_globals.shape)
        #         print("collected globals ", collected_globals)

        updated_globals = self._global_model(collected_globals)

        graph.update({'graph_globals': torch.squeeze(updated_globals, dim=0)})
        #         print("global block ", graph.graph_globals.shape)
        return graph