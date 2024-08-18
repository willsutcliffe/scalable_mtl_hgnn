from torch_geometric.nn.models import MLP
import torch
import torch.nn as nn
import tree
from wmpgnn.gnn.graph_network import GraphNetwork
from wmpgnn.gnn.graphcoder import GraphIndependent
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

        self._network = GraphNetwork(edge_model=make_mlp(edge_output_size),
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
                 global_op=None):
        super(EncodeProcessDecode, self).__init__()
        self._encoder = MLPGraphIndependent(mlp_output_size, mlp_output_size, mlp_output_size)
        self._core1 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
        self._core2 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
        self._core3 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
        self._core4 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
        self._core5 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
        self._core6 = MLPGraphNetwork(edge_output_size=mlp_output_size, node_output_size=mlp_output_size,
                                      global_output_size=mlp_output_size)
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
        #         latent0 = latent
        latent1 = self._core1(latent)
        core_input = graph_concat([latent0, latent1], axis=1)
        if self._core1._network.prune == True:
            indices = self._core1._network.indices
            updated_edges = latent0.edges[indices, :]
            updated_senders = latent0.senders[indices]
            updated_receivers = latent0.receivers[indices]
            updated_edge_pos = latent0.edgepos[indices]
            updated_y = latent0.y[indices]
            latent0.update({'edges': updated_edges,
                            'senders': updated_senders,
                            'receivers': updated_receivers,
                            'edgepos': updated_edge_pos,
                            'y': updated_y})
        latent2 = self._core2(core_input)
        if self._core2._network.prune == True:
            indices = self._core2._network.indices
            updated_edges = latent0.edges[indices, :]
            updated_senders = latent0.senders[indices]
            updated_receivers = latent0.receivers[indices]
            updated_edge_pos = latent0.edgepos[indices]
            updated_y = latent0.y[indices]
            latent0.update({'edges': updated_edges,
                            'senders': updated_senders,
                            'receivers': updated_receivers,
                            'edgepos': updated_edge_pos,
                            'y': updated_y})
            # latent2 = self._core2(latent1)
        core_input = graph_concat([latent0, latent2], axis=1)
        latent3 = self._core3(core_input)
        # latent3 = self._core3(latent2)
        core_input = graph_concat([latent0, latent3], axis=1)
        latent4 = self._core4(core_input)
        core_input = graph_concat([latent0, latent4], axis=1)
        # latent4 = self._core4(latent3)
        latent5 = self._core5(core_input)
        core_input = graph_concat([latent0, latent5], axis=1)
        latent6 = self._core6(latent5)
        # latent5 = self._core5(latent4)
        # latent6 = self._core6(latent5)
        # print("Laten 0 nodes ", latent0.nodes)
        #         output_ops = []
        #         for _ in range(num_processing_steps):

        #         core_input = graph_concat([latent0, latent], axis=1)
        #             if _ ==0:
        #                 latent = self._core1(core_input)
        #             else:
        #                 latent = self._core2(core_input)
        decoded_op = self._decoder(latent6)
        # decoded_op = self._decoder(latent4)

        output = (self._output_transform(decoded_op))

        return output