from wmpgnn.blocks.abstract_module import AbstractModule
import torch

def no_transform():
    return lambda x: x


class WrappedModelFnModule(AbstractModule):
    def __init__(self, model_fn):
        super(WrappedModelFnModule, self).__init__()
        with self._enter_variable_scope():
            self._model = model_fn()

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class HeteroGraphCoder(AbstractModule):
    def __init__(self, node_types: list, edge_types: list,
                 edge_models=None, node_models=None, global_model=None, endecoder=True):
        super(HeteroGraphCoder, self).__init__()
        with self._enter_variable_scope():
            self._edge_types = edge_types
            self._node_types = node_types
            self._edge_models = {}
            self._node_models = {}
            self._endecoder = endecoder


            for node_type in self._node_types:
                self._node_models[node_type] = WrappedModelFnModule(node_models[node_type])

            for edge_type in self._edge_types:
                self._edge_models[edge_type] = WrappedModelFnModule(edge_models[edge_type])

            if global_model is None:
                self._global_model = lambda x: x
            else:
                self._global_model = WrappedModelFnModule(global_model)
            self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_models.items()})
            self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_models.items()})

    def forward(self, graph):
        if self._endecoder:
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x, graph[node_type].batch)
            for edge_type in self._edge_types:

                # edge_type = ('chargedtree', 'to', 'neutrals')
                # edge_src_idx = graph[edge_type].edge_index[0]
                # batch_src = graph['chargedtree'].batch

                # print(f"\n[GRAPH DEBUG]")
                # print(f"graph['chargedtree'].x.shape = {graph['chargedtree'].x.shape}")
                # print(f"graph['chargedtree'].batch.shape = {graph['chargedtree'].batch.shape}")
                # print(f"graph[edge_type].edge_index.shape = {graph[edge_type].edge_index.shape}")
                # print(f"edge_src_idx.max() = {edge_src_idx.max()}")
                # print(f"batch_src.shape = {batch_src.shape}")
                # print(f"BAD INDICES (edge_src_idx >= batch_src.shape[0]):")
                # print(edge_src_idx[edge_src_idx >= batch_src.shape[0]])

                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0]])
        else:
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x)
            for edge_type in self._edge_types:
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges)
        graph['globals'].x = self._global_model(graph['globals'].x)
        return graph