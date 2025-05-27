
from torch_geometric.nn.models import MLP
import torch
import torch.nn as nn
from wmpgnn.gnn.hetero_graph_network import HeteroGraphNetwork
from wmpgnn.gnn.hetero_graphcoder import HeteroGraphCoder
from wmpgnn.gnn.hetero_graph_network import edge_pruning, node_pruning

def make_mlp(output_size, hidden_channels=128, num_layers=4, norm="batch_norm"):
    """
    Create a factory function for a Multi-Layer Perceptron (MLP) with specified architecture.

    Args:
        output_size (int): Dimension of the output feature vector.
        hidden_channels (int, optional): Number of hidden units in each hidden layer. Defaults to 128.
        num_layers (int, optional): Total number of layers (including input and output). Defaults to 4.
        norm (str, optional): Normalization type, e.g., "batch_norm", "graph_norm". Defaults to "batch_norm".

    Returns:
        Callable[[], MLP]: A function that constructs an MLP when called.
    """
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
              out_channels=output_size, num_layers=num_layers, norm=norm)

def hetero_graph_concat(g1,g2):
    """
    Concatenate two heterogeneous graphs along their edge and node feature dimensions.

    This function clones the first graph, then for each edge and node type,
    concatenates the corresponding features from the second graph along the last dimension.
    Global features are also concatenated similarly.

    Args:
        g1 (HeteroData): The first heterogeneous graph data.
        g2 (HeteroData): The second heterogeneous graph data, to be concatenated with g1.

    Returns:
        HeteroData: A new heterogeneous graph with concatenated features.
    """
    graph = g1.clone()
    for edge_type in g1.edge_types:
        graph[edge_type].edges = torch.cat( [g1[edge_type].edges, g2[edge_type].edges], -1)
    for node_type in g1.node_types:
        graph[node_type].x = torch.cat( [g1[node_type].x, g2[node_type].x], -1)
    graph['globals'].x = torch.cat( [g1['globals'].x, g2['globals'].x], -1)
    return graph


class HeteroGNN(nn.Module):
    """
    A modular heterogeneous graph neural network (GNN) architecture for multi-relational graphs.

    Consists of an encoder, multiple message-passing blocks, and a decoder.
    Supports edge and node feature transformations, as well as optional pruning.
    """
    def __init__(self,
                 node_types,
                 edge_types,
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
                 weighted_mp=False,
                 norm="batch_norm"):
        """
        Initializes the HeteroGNN model with encoders, message-passing blocks, and decoders.

        Args:
            node_types (List[str]): List of node type identifiers.
            edge_types (List[Tuple[str, str, str]]): List of edge type triplets (src, relation, dst).
            mlp_output_size (int): Dimensionality of the MLP output used in each model component.
            edge_op (int, optional): Output dimensionality for edge-level prediction heads.
            node_op (int, optional): Output dimensionality for node-level prediction heads.
            global_op (int, optional): Output dimensionality for graph-level prediction heads.
            num_blocks (int, optional): Number of HeteroGraphNetwork message-passing blocks. Default is 2.
            use_edge_weights (bool, optional): Whether to use edge weights in message passing. Default is True.
            use_node_weights (bool, optional): Whether to use node weights in message passing. Default is True.
            mlp_channels (int, optional): Hidden dimension for internal MLPs. Default is 128.
            mlp_layers (int, optional): Number of layers in internal MLPs. Default is 4.
            weight_mlp_channels (int, optional): Hidden dimension for edge/node weight MLPs. Default is 16.
            weight_mlp_layers (int, optional): Number of layers in edge/node weight MLPs. Default is 4.
            weighted_mp (bool, optional): Whether to use weighted message passing. Default is False.
            norm (str, optional): Normalization type to use in MLPs. Default is "batch_norm".
        """
        super(HeteroGNN, self).__init__()

        self.edge_types = edge_types
        self.node_types = node_types
        mlp = make_mlp(mlp_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm)
        if norm == "graph_norm":
            global_mlp = make_mlp(mlp_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm="batch_norm")
        else:
            global_mlp = make_mlp(mlp_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers, norm=norm)
        self._encoder = HeteroGraphCoder(node_types, edge_types,
                                         edge_models={edge_type: mlp for edge_type in edge_types},
                                         node_models={node_type: mlp for node_type in node_types}, global_model=global_mlp)

        self._blocks = []
        for i in range(num_blocks):
            self._blocks.append(
                HeteroGraphNetwork(node_types, edge_types, edge_model=mlp, node_model=mlp, global_model=mlp,
                                   use_node_weights=use_node_weights, use_edge_weights=use_edge_weights,
                                   weight_mlp_channels=weight_mlp_channels, weight_mlp_layers=weight_mlp_layers,
                                   weighted_mp=weighted_mp, norm=norm))
        self._blocks = nn.ModuleList(self._blocks)

        self._decoder = HeteroGraphCoder(node_types, edge_types,
                                         edge_models={edge_type: mlp for edge_type in edge_types},
                                         node_models={node_type: mlp for node_type in node_types}, global_model=global_mlp)

        # Transforms the outputs into the appropriate shapes.
        def no_transform():
            return lambda x: x

        if edge_op is None:
            edge_fn = None
        else:
            edge_fn = lambda: nn.Linear(mlp_output_size, edge_op)
        if node_op is None:

            node_fn = {node_type: no_transform for node_type in node_types}
        else:
            node_fn = lambda: nn.Linear(mlp_output_size, node_op)
        if global_op is None:
            global_fn = None
        else:
            global_fn = lambda: nn.Linear(mlp_output_size, global_op)

        edge_models = {('tracks',
                        'to',
                        'pvs'): lambda: nn.Linear(mlp_output_size, 1), ('tracks',
                                                                        'to',
                                                                        'tracks'): lambda: nn.Linear(mlp_output_size,
                                                                                                     4)}

        self._output_transform = HeteroGraphCoder(node_types, edge_types, edge_models=edge_models,
                                                  node_models=node_fn, global_model=global_fn, endecoder=False)



    def forward(self, input_op):
        """
        Applies the heterogeneous GNN to the input graph.

        Args:
            input_op (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.

        Returns:
            HeteroData: Output graph with transformed node, edge, and global features (optionally projected).
        """

        latent = self._encoder(input_op)
        latent0 = latent.clone()

        for b, core in enumerate(self._blocks):
            latent = core(latent)
            if core.edge_prune == True:
                for edge_type in self.edge_types:
                    if edge_type == ('tracks', 'to', 'tracks'):
                        edge_indices = core.edge_indices[edge_type]
                        edge_pruning(edge_indices, latent0, edge_type)
            if core.node_prune == True:
                for node_type in self.node_types:
                    if node_type == "tracks":
                        node_indices = core.node_indices['tracks']
                        node_pruning(node_indices, latent0, node_type, [('tracks', 'to', 'tracks')], device=core.device)
            if b < (len(self._blocks) - 1):
                core_input = hetero_graph_concat(latent0, latent)
                latent = core_input


        decoded_op = self._decoder(latent)

        output = (self._output_transform(decoded_op))

        return output