from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_aggregators import HeteroEdgesToGlobalsAggregator
from wmpgnn.blocks.hetero_aggregators import HeteroNodesToGlobalsAggregator
import torch

class HeteroGlobalBlock(AbstractModule):


    def __init__(self, node_types, edge_types, global_model_fn, use_edges=True,
                 use_nodes=True, use_globals=True, weighted_mp=False):
        super(HeteroGlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals
        self._node_types = node_types
        self._edge_types = edge_types

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = HeteroEdgesToGlobalsAggregator(weighted=weighted_mp)

            if self._use_nodes:
                self._nodes_aggregator = HeteroNodesToGlobalsAggregator(weighted=weighted_mp)



    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        #         if global_model_kwargs is None:
        #             global_model_kwargs = {}

        globals_to_collect = []

        if self._use_edges:
            for edge_type in self._edge_types:
                globals_to_collect.append(self._edges_aggregator(graph, edge_type, edge_weights[edge_type]))

        if self._use_nodes:
            for node_type in self._node_types:
                globals_to_collect.append(self._nodes_aggregator(graph, node_type, node_weights[node_type]))

        if self._use_globals:
            globals_to_collect.append(graph['globals'].x)

        # for col in globals_to_collect:
        #     print(' in global block ', torch.isnan(col).any())
        collected_globals = torch.cat(globals_to_collect, axis=-1)




        updated_globals = self._global_model(collected_globals)

        graph['globals'].x = updated_globals


        return graph