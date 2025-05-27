from wmpgnn.blocks.abstract_module import AbstractModule
import contextlib

class WrappedModelFnModule(AbstractModule):
    """
    Wraps a no-argument model‑construction function into an AbstractModule.

    This allows you to pass in a function that builds and returns a submodule
    (e.g. a Keras or PyTorch model), and have it integrated into the variable
    scope management of the parent framework.

    Args:
        model_fn (callable[[], AbstractModule]): A zero-argument function that,
            when called, constructs and returns an `AbstractModule` instance.
    """
    def __init__(self, model_fn):
        """
        Initializes the WrappedModelFnModule.

        Enters the module's variable scope and constructs the wrapped model by
        calling the provided factory function.

        Args:
            model_fn (callable[[], AbstractModule]): A no-argument callable
                that returns an `AbstractModule` instance when invoked.
        """
        super(WrappedModelFnModule, self).__init__()
        with self._enter_variable_scope():
            self._model = model_fn()

    def forward(self, *args, **kwargs):
        """
        Calls the wrapped model’s forward method with the provided inputs.

        Args:
            *args: Positional arguments to pass to the wrapped model.
            **kwargs: Keyword arguments to pass to the wrapped model.

        Returns:
            The output of `self._model(*args, **kwargs)`.
        """
        return self._model(*args, **kwargs)


class GraphIndependent(AbstractModule):
    """
    Applies independent transformations to the edges, nodes, and global features
    of a graph.

    In encoder mode, each transform function (edge_model, node_model, global_model)
    receives both the feature tensor and an auxiliary positional tensor
    (e.g. `edgepos`, `batch`, etc.). In decoder mode, only the feature tensor
    is passed through.

    Args:
        edge_model (callable[[], AbstractModule] or None):
            Zero-arg constructor for the edge update model. If None, edges
            pass through unchanged (identity).
        node_model (callable[[], AbstractModule] or None):
            Zero-arg constructor for the node update model. If None, nodes
            pass through unchanged.
        global_model (callable[[], AbstractModule] or None):
            Zero-arg constructor for the global update model. If None,
            globals pass through unchanged.
        encoder (bool):
            If True, call models as `model(features, vars)` so that positional/
            index information can be used. If False, call models as
            `model(features)` only.

    Attributes:
        _edge_model (Callable): Either a `WrappedModelFnModule` or identity.
        _node_model (Callable): Either a `WrappedModelFnModule` or identity.
        _global_model (Callable): Either a `WrappedModelFnModule` or identity.
        _encoder (bool): Flag for encoder vs. decoder behavior.
    """
    def __init__(self, edge_model=None, node_model=None, global_model=None, encoder=True):
        """
        Initializes the GraphIndependent module.

        Wraps each provided model factory in a `WrappedModelFnModule`, or uses
        the identity function if no factory is provided. All submodules are
        created within this module's variable scope.

        Args:
            edge_model (callable[[], AbstractModule] or None):
                Factory for the edge update module. If None, edges are unchanged.
            node_model (callable[[], AbstractModule] or None):
                Factory for the node update module. If None, nodes are unchanged.
            global_model (callable[[], AbstractModule] or None):
                Factory for the global update module. If None, globals are unchanged.
            encoder (bool):
                Whether to include positional/index data when calling submodules.
        """
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


    def forward(self, graph):
        """
        Performs independent edge, node, and global updates for encoder and decoder
        updates.

        The graph is mutated in place: its `edges`, `nodes`, and `graph_globals`
        fields are overwritten with the model outputs.

         Args:
            graph: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Graph: The same graph object, now with updated features.
        """
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
