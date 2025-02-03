
from torch_geometric.nn.models import MLP
import torch
import torch.nn as nn
import tree
from wmpgnn.gnn.hetero_graph_network import HeteroGraphNetwork
from wmpgnn.gnn.hetero_graphcoder import HeteroGraphCoder
from wmpgnn.gnn.hetero_graph_network import edge_pruning

def make_mlp(output_size, hidden_channels=128, num_layers=4):
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
              out_channels=output_size, num_layers=num_layers , norm=None)

def hetero_graph_concat(g1,g2):
    graph = g1.clone()
    for edge_type in g1.edge_types:
        graph[edge_type].edges = torch.cat( [g1[edge_type].edges, g2[edge_type].edges], -1)
    for node_type in g1.node_types:
        graph[node_type].x = torch.cat( [g1[node_type].x, g2[node_type].x], -1)
    graph['globals'].x = torch.cat( [g1['globals'].x, g2['globals'].x], -1)
    return graph


class HeteroGNN(nn.Module):
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
                 weighted_mp=False):
        super(HeteroGNN, self).__init__()
        mlp = make_mlp(mlp_output_size, hidden_channels=mlp_channels, num_layers=mlp_layers)
        self._encoder = HeteroGraphCoder(node_types, edge_types,
                                         edge_models={edge_type: mlp for edge_type in edge_types},
                                         node_models={node_type: mlp for node_type in node_types}, global_model=mlp)

        self._blocks = []
        for i in range(num_blocks):
            self._blocks.append(
                HeteroGraphNetwork(node_types, edge_types, edge_model=mlp, node_model=mlp, global_model=mlp,
                                   use_node_weights=use_node_weights, use_edge_weights=use_edge_weights,
                                   weight_mlp_channels=weight_mlp_channels, weight_mlp_layers=weight_mlp_layers,
                                   weighted_mp=weighted_mp))
        self._blocks = nn.ModuleList(self._blocks)

        self._decoder = HeteroGraphCoder(node_types, edge_types,
                                         edge_models={edge_type: mlp for edge_type in edge_types},
                                         node_models={node_type: mlp for node_type in node_types}, global_model=mlp)

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
        # edge_models ={('tracks',
        #                 'to',
        #                 'tracks') : lambda: nn.Linear(mlp_output_size, 4).cuda()}
        self._output_transform = HeteroGraphCoder(node_types, edge_types, edge_models=edge_models,
                                                  node_models=node_fn, global_model=global_fn)



    def forward(self, input_op):

        latent = self._encoder(input_op)
        latent0 = latent.clone()

        for b, core in enumerate(self._blocks):
            latent = core(latent)
            if core.edge_prune == True:
                edge_indices = core.edge_indices[('tracks', 'to', 'tracks')]
                edge_pruning(edge_indices, latent0, ('tracks', 'to', 'tracks'))
            if b < (len(self._blocks) - 1):
                core_input = hetero_graph_concat(latent0, latent)
                latent = core_input


        decoded_op = self._decoder(latent)

        output = (self._output_transform(decoded_op))

        return output