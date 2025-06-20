from wmpgnn.blocks.abstract_module import AbstractModule
import torch



def no_transform():
    """
    Identity transform factory.

    Returns:
        callable: A function that returns its input unchanged.
    """
    return lambda x: x


class WrappedModelFnModule(AbstractModule):
    """
    Adapter module that wraps a zero-argument model factory function into
    an `AbstractModule`, enabling it to participate in Sonnet-style
    variable scoping and PyTorch module hierarchies.

    It simply constructs the user‑provided model once during initialization
    and delegates all `forward` calls to it.

    Attributes:
        _model (nn.Module): The wrapped model instance created by `model_fn()`.
    """
    def __init__(self, model_fn):
        """
        Initialize the wrapper by constructing the inner model.

        Args:
            model_fn (callable): Zero-argument function that returns an `nn.Module`.
        """
        
        super(WrappedModelFnModule, self).__init__()
        
        with self._enter_variable_scope():
            self._model = model_fn()

    def forward(self, *args, **kwargs):
        """
        Forward pass that delegates to the wrapped model.

        Accepts arbitrary positional and keyword arguments to match the
        signature of the underlying `nn.Module`.

        Returns:
            The result of `self._model(*args, **kwargs)`.
        """
        return self._model(*args, **kwargs)


class HeteroGraphCoder(AbstractModule):
    """
    Encoder/decoder for heterogeneous graphs that applies separate
    transformations to each node type, edge type, and the global features.

    Depending on the `endecoder` flag, it can operate as:
      - An **encoder**: transforms node and edge features without batching info.
      - A **decoder**: transforms node and edge features using per-node or
        per-edge batch indices for graph-level operations.

    The global feature transformation is always applied to `graph['globals'].x`.

    Attributes:
        _node_types (List[str]): Keys for node feature sets in the graph.
        _edge_types (List[tuple]): Keys for edge relations (src_type, dst_type, rel_key).
        _node_models (dict): Maps node_type to its `WrappedModelFnModule`.
        _edge_models (dict): Maps edge_type to its `WrappedModelFnModule`.
        _global_model (callable or WrappedModelFnModule): Model transforming global features.
        _endecoder (bool): If True, node/edge transforms receive batch indices.
        _node_models_model_dict (ModuleDict): PyTorch container of node models.
        _edge_models_model_dict (ModuleDict): PyTorch container of edge models.
    """
    def __init__(self, node_types: list, edge_types: list,
                 edge_models=None, node_models=None, global_model=None, endecoder=True):
        """
        Initialize the heterogeneous graph coder.

        Args:
            node_types (List[str]): List of node-set keys present in the graph.
            edge_types (List[tuple]): List of edge-type identifiers as
                `(src_node_type, dst_node_type, edge_key)`.
            edge_models (dict): Maps each edge_type to a zero-argument function
                returning an `nn.Module` for transforming edge features.
            node_models (dict): Maps each node_type to a zero-argument function
                returning an `nn.Module` for transforming node features.
            global_model (callable, optional): Zero-argument function returning
                an `nn.Module` to transform global features. If None, no-op.
            endecoder (bool): If True, passes `(features, batch_idx)` to each
                model; if False, passes only `features`.
        """
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
        """
        Apply per-type transformations to nodes, edges, and globals.

        If `endecoder=True`, each node model is called as
        `model(node_features)` and each edge model as
        `model(edge_features)`. Otherwise, models receive
        only the feature tensors.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.

        Returns:
            HeteroData: The same graph object with updated `.x` and `.edges`
            for each type, and updated global features in `graph['globals'].x`.
        """
        if self._endecoder:
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x, graph[node_type].batch)
            for edge_type in self._edge_types:
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges, graph[edge_type[0]].batch[ graph[edge_type].edge_index[0]]) # non-deterministic behaviour: see https://docs.nvidia.com/cuda/cublas/index.html#results-reproducibility
        else:
            for node_type in self._node_types:
                graph[node_type].x = self._node_models[node_type](graph[node_type].x)
            for edge_type in self._edge_types:
                graph[edge_type].edges = self._edge_models[edge_type](graph[edge_type].edges)
        graph['globals'].x = self._global_model(graph['globals'].x)
        return graph