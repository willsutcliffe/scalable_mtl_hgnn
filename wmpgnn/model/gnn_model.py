from torch_geometric.nn.models import MLP
import torch
import torch.nn as nn
import tree
from wmpgnn.gnn.graph_network import GraphNetwork
from wmpgnn.gnn.graphcoder import GraphIndependent
from wmpgnn.gnn.graph_network import edge_pruning, node_pruning
NUM_LAYERS = 4
HIDDEN_CHANNELS=128


def make_mlp(output_size):
    return lambda: MLP(in_channels=-1, hidden_channels=HIDDEN_CHANNELS,
              out_channels=output_size, num_layers=NUM_LAYERS , norm=None)


def _nested_concatenate(input_graphs, field_name, axis):
    features_list = [getattr(gr, field_name) for gr in input_graphs
                     if getattr(gr, field_name) is not None]
    if not features_list:
        return None

    if len(features_list) < len(input_graphs):
        raise ValueError(
            "All graphs or no graphs must contain {} features.".format(field_name))

    # if field_name=="graph_globals":
    #    return tree.map_structure(lambda *x: torch.cat(x, axis-1), *features_list)
    # else:
    return tree.map_structure(lambda *x: torch.cat(x, axis), *features_list)


def graph_concat(input_graphs, axis):
    if not input_graphs:
        raise ValueError("List argument `input_graphs` is empty")
    if len(input_graphs) == 1:
        return input_graphs[0]

    nodes = _nested_concatenate(input_graphs, "nodes", axis)
    edges = _nested_concatenate(input_graphs, "edges", axis)
    graph_globals = _nested_concatenate(input_graphs, "graph_globals", axis)
    #     print(graph_globals)
    graph = input_graphs[0].clone()
    output = graph.update({'nodes': nodes, 'edges': edges, 'graph_globals': graph_globals})
    if axis != 0:
        output
        return graph

    else:
        raise ValueError("axis is 0")



class MLPGraphNetwork(nn.Module):
    def __init__(self, edge_output_size, node_output_size, global_output_size):
        super(MLPGraphNetwork, self).__init__()

        self._network = GraphNetwork(
                                     edge_model=make_mlp(edge_output_size),
                                     node_model=make_mlp(node_output_size),
                                     use_globals=True,
                                     global_model=make_mlp(node_output_size))
        # global_model=make_mlp(global_output_size))

    def forward(self, inputs):
        return self._network(inputs)


class MLPGraphIndependent(nn.Module):
    def __init__(self, edge_output_size, node_output_size, global_output_size):
        super(MLPGraphIndependent, self).__init__()

        self._network = GraphIndependent(edge_model=make_mlp(edge_output_size),
                                         node_model=make_mlp(node_output_size),
                                         global_model=make_mlp(node_output_size))

    def forward(self, inputs):
        return self._network(inputs)


class EncodeProcessDecode(nn.Module):
    def __init__(self,
                 mlp_output_size,
                 edge_op=None,
                 node_op=None,
                 global_op=None,
                 num_blocks=2):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = MLPGraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size)

        self._blocks = []
        for i in range(num_blocks):
            self._core = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                     global_output_size=mlp_output_size)
            self._blocks.append(self._core)
        self._blocks = nn.ModuleList(self._blocks)

        self._decoder = MLPGraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size)

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

        self._output_transform = GraphIndependent(edge_fn, node_fn, global_fn)

    #         self._encoder2 = GraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size)

    def forward(self, input_op, num_processing_steps):
        # print("input ", input_op.graph_globals.shape)
        latent = self._encoder(input_op)
        latent0 = latent.clone()
        for b, core in enumerate(self._blocks):
            #print("latent ", latent.edgepos)
            latent = core(latent)
            if core._network.edge_prune == True:
                edge_indices = core._network.edge_indices

                edge_pruning(edge_indices, latent0)

            if core._network.node_prune == True:
                node_indices = core._network.node_indices
                node_pruning(node_indices, latent0)
            if b < (len(self._blocks)-1):
                core_input = graph_concat([latent0, latent], axis=1)

                latent = core_input

            else:
                pass
        decoded_op = self._decoder(latent)

        output = (self._output_transform(decoded_op))

        return output