from torch_geometric.nn.models import MLP
import torch
import torch.nn as nn
import tree
from wmpgnn.gnn.graph_network import GraphNetwork
from wmpgnn.gnn.graphcoder import GraphIndependent
from wmpgnn.gnn.graph_network import edge_pruning, node_pruning, node_pruning2


def make_mlp(output_size, hidden_channels=128, num_layers=4, norm="batch_norm"):
    """
    Factory for creating an MLP that maps from arbitrary input size to `output_size`.

    Args:
        output_size (int): Number of output channels of the final linear layer.
        hidden_channels (int): Number of hidden units in each intermediate layer.
        num_layers (int): Total number of linear layers (including input and output).
        norm (str): Normalization type to use between layers (e.g. "batch_norm").

    Returns:
        Callable[[], MLP]: A zero-argument constructor for a PyG `MLP` with the given config.
    """
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
          out_channels=output_size, num_layers=num_layers, norm=norm) # , norm='batch_norm')


def _nested_concatenate(input_graphs, field_name, axis):
    """
    Concatenate a named tensor field across a list of graph objects, preserving structure.

    Args:
        input_graphs (List): Sequence of graph-like objects.
        field_name (str): Attribute name to extract (e.g. "nodes", "edges", "graph_globals").
        axis (int): Dimension along which to concatenate.

    Returns:
        Tensor or nested structure of Tensors: Concatenated values for that field.

    Raises:
        ValueError: If some but not all graphs are missing the specified field.
    """
    features_list = [getattr(gr, field_name) for gr in input_graphs
                     if getattr(gr, field_name) is not None]
    if not features_list:
        return None

    if len(features_list) < len(input_graphs):
        raise ValueError(
            "All graphs or no graphs must contain {} features.".format(field_name))

    return tree.map_structure(lambda *x: torch.cat(x, axis), *features_list)


def graph_concat(input_graphs, axis):
    """
    Concatenate the node, edge, and global feature fields of multiple graphs.

    Args:
        input_graphs (List): List of graph objects (must support `.clone()` and `.update()`).
        axis (int): Dimension along which to concatenate the feature tensors.

    Returns:
        Graph: A new graph whose features are the concatenation of inputs.

    Raises:
        ValueError: If the input list is empty or `axis == 0` (not supported).
    """
    if not input_graphs:
        raise ValueError("List argument `input_graphs` is empty")
    if len(input_graphs) == 1:
        return input_graphs[0]

    nodes = _nested_concatenate(input_graphs, "nodes", axis)
    edges = _nested_concatenate(input_graphs, "edges", axis)
    graph_globals = _nested_concatenate(input_graphs, "graph_globals", axis)

    graph = input_graphs[0].clone()
    output = graph.update({'nodes': nodes, 'edges': edges, 'graph_globals': graph_globals})
    if axis != 0:
        output
        return graph

    else:
        raise ValueError("axis is 0")



class MLPGraphNetwork(nn.Module):
    """
    Graph network composed of: encoder → multiple MLP‐based GN blocks → decoder.

    Uses `GraphNetwork` under the hood, with MLP models for edges, nodes, and globals.

    Args:
        edge_output_size (int): Output dim for edge MLP.
        node_output_size (int): Output dim for node MLP.
        global_output_size (int): Output dim for global MLP.
        use_edge_weights (bool): Whether to learn/apply per-edge weights.
        use_node_weights (bool): Whether to learn/apply per-node weights.
        mlp_channels (int): Hidden size for the internal MLPs.
        mlp_layers (int): Number of layers in the internal MLPs.
        weight_mlp_channels (int): Hidden size for the weighting MLPs.
        weight_mlp_layers (int): Number of layers for the weighting MLPs.
        weighted_mp (bool): Whether to pass weights into message-passing.
        norm (str): Normalization type for MLPs.
    """

    def __init__(self, edge_output_size, node_output_size, global_output_size,
                 use_edge_weights, use_node_weights, mlp_channels=128, mlp_layers=4,
                 weight_mlp_channels=16, weight_mlp_layers=4, weighted_mp = False,
                 norm = "batch_norm"):
        """
        Initialize the MLPGraphNetwork wrapper around a `GraphNetwork`.

        Constructs an internal `GraphNetwork` where the edge, node, and global update
        functions are each small MLPs created by `make_mlp`. Also configures whether
        per-edge and per-node weights are learned and passed into message passing.

        Args:
            edge_output_size (int): Dimensionality of the edge MLP outputs.
            node_output_size (int): Dimensionality of the node MLP outputs.
            global_output_size (int): Dimensionality of the global MLP outputs.
            use_edge_weights (bool): If True, learn and apply scalar weights to edges
                before node and global updates.
            use_node_weights (bool): If True, learn and apply scalar weights to nodes
                before the global update.
            mlp_channels (int, optional): Hidden size for the edge/node/global MLPs.
                Defaults to 128.
            mlp_layers (int, optional): Number of layers in each edge/node/global MLP.
                Defaults to 4.
            weight_mlp_channels (int, optional): Hidden size for the weight-prediction
                MLPs used to compute per-edge and per-node weights. Defaults to 16.
            weight_mlp_layers (int, optional): Number of layers in the weight-prediction
                MLPs. Defaults to 4.
            weighted_mp (bool, optional): If True, passes the computed weights into the
                underlying message-passing aggregators. Defaults to False.
            norm (str, optional): Normalization type for all MLPs (e.g., "batch_norm",
                "layer_norm"). If set to "graph_norm", global MLP will use "batch_norm"
                instead. Defaults to "batch_norm".
        """
        super(MLPGraphNetwork, self).__init__()
        if norm != "graph_norm":
            global_norm = norm
        else:
            global_norm = "batch_norm"
        self._network = GraphNetwork(
                                     edge_model=make_mlp(edge_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm),
                                     node_model=make_mlp(node_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm),
                                     use_globals=True,
                                     global_model=make_mlp(node_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=global_norm),
                                     use_edge_weights=use_edge_weights,
                                     use_node_weights=use_node_weights,
                                     weight_mlp_layers=weight_mlp_layers,
                                     weight_mlp_channels=weight_mlp_channels,
                                    weighted_mp=weighted_mp,
                                    norm = norm)
        # global_model=make_mlp(global_output_size))

    def forward(self, inputs):
        """
        Forward pass through the wrapped GraphNetwork.

        Args:
            inputs: A PyG `Data` object with `edges`, `nodes`, `graph_globals`, etc.

        Returns:
            Data: The updated graph with new features.
        """
        return self._network(inputs)


class MLPGraphIndependent(nn.Module):
    """
    Applies separate MLP transforms to edges, nodes, and global features without message passing.

    This module wraps a `GraphIndependent` network from `wmpgnn.gnn.graphcoder`, using
    MLPs for each graph component. It can operate as an encoder or decoder depending on
    the `encoder` flag.

    Attributes:
        _network (GraphIndependent): Underlying network applying independent transforms.
    """
    def __init__(self, edge_output_size, node_output_size, global_output_size,
                 mlp_channels=128, mlp_layers=4, norm = "batch_norm", encoder=True):
        """
        Initialize the independent MLP graph transformer.

        Args:
            edge_output_size (int): Output feature dimension for edges.
            node_output_size (int): Output feature dimension for nodes.
            global_output_size (int): Output feature dimension for globals.
            mlp_channels (int): Hidden size in each MLP layer.
            mlp_layers (int): Number of layers in each MLP.
            norm (str): Normalization type for MLPs ("batch_norm", "layer_norm", etc.).
            encoder (bool): If True, MLPs receive `(features, batch_idx)`; else only `features`.
        """
        super(MLPGraphIndependent, self).__init__()
        if norm != "graph_norm":
            global_norm = norm
        else:
            global_norm = "batch_norm"

        self._network = GraphIndependent(edge_model=make_mlp(edge_output_size , hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm),
                                         node_model=make_mlp(node_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm),
                                         global_model=make_mlp(node_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=global_norm),
                                         encoder=encoder)

    def forward(self, inputs):
        """
        Apply independent MLP transforms to each graph component.

         Args:
            inputs: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Graph or HeteroData with updated features.
        """
        return self._network(inputs)


class GNN(nn.Module):
    """
    Stacked GraphNetwork blocks with an optional encoder and decoder, plus
    configurable pruning at each block.

    Pipeline:
      1. Encode via `MLPGraphIndependent`.
      2. Apply N `MLPGraphNetwork` blocks, with optional pruning after each.
      3. Optionally concatenate intermediate and updated features.
      4. Decode via `MLPGraphIndependent`.
      5. Final output transform via `GraphIndependent`.
    """
    def __init__(self,
                 mlp_output_size,
                 edge_op=None,
                 node_op=None,
                 global_op=None,
                 num_blocks=2,
                 use_edge_weights=True,
                 use_node_weights=True,
                 mlp_channels=128,
                 mlp_layers=4,
                 weight_mlp_channels=16,
                 weight_mlp_layers=4,
                 weighted_mp = False,
                 norm = "batch_norm"):
        """
        Initializes the GNN with an encoder, stacked MLPGraphNetwork blocks, decoder,
        and optional output projections for edge, node, and global features.

        Args:
            mlp_output_size (int): Output dimensionality of each MLP used in the encoder,
                core blocks, and decoder.
            edge_op (Optional[int]): Output dimensionality for the edge-level projection.
                If None, no edge output projection is applied.
            node_op (Optional[int]): Output dimensionality for the node-level projection.
                If None, no node output projection is applied.
            global_op (Optional[int]): Output dimensionality for the global-level projection.
                If None, no global output projection is applied.
            num_blocks (int): Number of MLPGraphNetwork message-passing blocks to apply.
            use_edge_weights (bool): Whether to compute and use learned edge weights during
                message passing.
            use_node_weights (bool): Whether to compute and use learned node weights during
                message passing.
            mlp_channels (int): Hidden dimensionality of the internal MLPs.
            mlp_layers (int): Number of layers in each MLP.
            weight_mlp_channels (int): Hidden dimensionality of the MLPs used to compute
                edge and node weights (if enabled).
            weight_mlp_layers (int): Number of layers in the edge/node weighting MLPs.
            weighted_mp (bool): Whether to apply weighted message passing using the
                learned edge and node weights.
            norm (str): Normalization strategy to use within MLPs (e.g., "batch_norm",
                "layer_norm", or None).
        """
        super(GNN, self).__init__()


        self._encoder = MLPGraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size,
                                            mlp_channels=mlp_channels, mlp_layers=mlp_layers, norm = norm)

        self._blocks = []
        for i in range(num_blocks):
            self._core = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                     global_output_size=mlp_output_size,use_edge_weights=use_edge_weights,
                                         use_node_weights=use_node_weights, mlp_channels=mlp_channels,
                                         mlp_layers=mlp_layers, weight_mlp_channels=weight_mlp_channels,
                                         weight_mlp_layers=weight_mlp_layers, weighted_mp=weighted_mp, norm = norm)
            self._blocks.append(self._core)
        self._blocks = nn.ModuleList(self._blocks)

        self._decoder = MLPGraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size,
                                            mlp_channels=mlp_channels, mlp_layers=mlp_layers, norm = norm)

        # Transforms the outputs into the appropriate shapes.
        if edge_op is None:
            edge_fn = None
        else:
            edge_fn = lambda: nn.Linear(mlp_output_size, edge_op)
        if node_op is None:
            node_fn = None
        else:
            node_fn = lambda: nn.Linear(mlp_output_size, node_op)
        if global_op is None:
            global_fn = None
        else:
            global_fn = lambda: nn.Linear(mlp_output_size, global_op)

        self._output_transform = GraphIndependent(edge_fn, node_fn, global_fn, encoder=False)


    def forward(self, input_op):
        """
        Full forward pass: encode, apply blocks with pruning, and decode.

         Args:
            input_op: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Updated graph with final feature transforms applied.
        """
        latent = self._encoder(input_op)
        latent0 = latent.clone()
        for b, core in enumerate(self._blocks):
            latent = core(latent)
            if core._network.edge_prune == True:
                edge_indices = core._network.edge_indices

                edge_pruning(edge_indices, latent0)

            if core._network.node_prune == True:
                node_indices = core._network.node_indices
                node_pruning(node_indices, latent0, core._network.device)
            if b < (len(self._blocks)-1):
                core_input = graph_concat([latent0, latent], axis=1)

                latent = core_input

            else:
                pass
        decoded_op = self._decoder(latent)

        output = (self._output_transform(decoded_op))

        return output