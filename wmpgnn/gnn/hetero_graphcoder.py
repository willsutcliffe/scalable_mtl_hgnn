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
                 edge_models=None, node_models=None, global_model=None):
        super(HeteroGraphCoder, self).__init__()
        with self._enter_variable_scope():
            self._edge_types = edge_types
            self._node_types = node_types
            self._edge_models = {}
            self._node_models = {}


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
        for node_type in self._node_types:
            graph[node_type].x = self._node_models[node_type](graph[node_type].x)
        for edge_type in self._edge_types:
            graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges)

        graph['globals'].x = self._global_model(graph['globals'].x)
        return graph