from wmpgnn.blocks.abstract_module import AbstractModule
import contextlib

class WrappedModelFnModule(AbstractModule):
    def __init__(self, model_fn):
        super(WrappedModelFnModule, self).__init__()
        with self._enter_variable_scope():
            self._model = model_fn()

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class GraphIndependent(AbstractModule):
    def __init__(self, edge_model=None, node_model=None, global_model=None, encoder=True):
        super(GraphIndependent, self).__init__()
        with self._enter_variable_scope():
            if edge_model is None:
                self._edge_model = lambda x: x
            else:
                self._edge_model = WrappedModelFnModule(edge_model)

            if node_model is None:
                self._node_model = lambda x: x
            else:
                self._node_model = WrappedModelFnModule(node_model)

            if global_model is None:
                self._global_model = lambda x: x
            else:
                self._global_model = WrappedModelFnModule(global_model)

        self._encoder = encoder

    #     def forward(self, graph, edge_model_kwargs=None, node_model_kwargs=None, global_model_kwargs=None):
    def forward(self, graph):

        if self._encoder:
            graph.update(
                {'edges': self._edge_model(graph.edges, graph.edgepos),
                 'nodes': self._node_model(graph.nodes, graph.batch),
                 'graph_globals': self._global_model(graph.graph_globals)})
        else:
            graph.update(
                {'edges': self._edge_model(graph.edges),
                 'nodes': self._node_model(graph.nodes),
                 'graph_globals': self._global_model(graph.graph_globals)})
        return graph
