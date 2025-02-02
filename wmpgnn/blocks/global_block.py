from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.aggregators import EdgesToGlobalsAggregator
from wmpgnn.blocks.aggregators import NodesToGlobalsAggregator
import torch
from torch_scatter import scatter_add
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_scatter import scatter_std

class GlobalBlock(AbstractModule):


    def __init__(self, global_model_fn, use_edges=True, use_nodes=True, use_globals=True, weighted_mp=False):
        super(GlobalBlock, self).__init__()

        self._use_edges = use_edges
        self._use_nodes = use_nodes
        self._use_globals = use_globals

        with self._enter_variable_scope():
            self._global_model = global_model_fn()
            if self._use_edges:
                self._edges_aggregator = EdgesToGlobalsAggregator(weighted=weighted_mp)

            if self._use_nodes:
                self._nodes_aggregator = NodesToGlobalsAggregator(weighted=weighted_mp)



    def forward(self, graph, edge_weights, node_weights, global_model_kwargs=None):
        #         if global_model_kwargs is None:
        #             global_model_kwargs = {}

        globals_to_collect = []

        if self._use_edges:
            globals_to_collect.append(self._edges_aggregator(graph, edge_weights))
            #globals_to_collect.append(self._edges_aggregator_max(graph, edge_weights)[0])
            #globals_to_collect.append(self._edges_aggregator_mean(graph, edge_weights))
            #globals_to_collect.append(self._edges_aggregator_std(graph, edge_weights))


        if self._use_nodes:
            globals_to_collect.append(self._nodes_aggregator(graph, node_weights))

        if self._use_globals:
            globals_to_collect.append(graph.graph_globals)


        collected_globals = torch.cat(globals_to_collect, axis=-1)




        updated_globals = self._global_model(collected_globals)

        graph.update({'graph_globals':  updated_globals})
        #graph.update({'graph_globals': torch.squeeze(updated_globals, dim=0)})

        return graph