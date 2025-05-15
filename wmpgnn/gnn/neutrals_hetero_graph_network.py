from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_edge_block  import HeteroEdgeBlock
from wmpgnn.blocks.hetero_global_block import HeteroGlobalBlock
from wmpgnn.blocks.hetero_node_block import HeteroNodeBlock
import torch
from torch_geometric.nn.models import MLP
from torch.nn import Linear, Sigmoid, Softmax
from torch_geometric.nn import knn
from torch_geometric.nn.pool.select import SelectTopK
from torch_scatter.composite import scatter_softmax
import contextlib


def weight_mlp(output_size, hidden_channels=16, num_layers=4, norm="batch_norm"):
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
                       out_channels=output_size, num_layers=num_layers, norm=norm)


def ones(device):
    return lambda x: torch.ones((x.shape[0], 1)).to(device)


def edge_pruning(edge_indices, graph, edge_type):
    graph[edge_type].edges = graph[edge_type].edges[edge_indices]
    graph[edge_type].edge_index = torch.vstack(
        [graph[edge_type].edge_index[0][edge_indices],
         graph[edge_type].edge_index[1][edge_indices]])
    graph[edge_type].y = graph[edge_type].y[edge_indices]


def node_pruning(node_indices, graph, node_type, edge_types, device = "cuda"):
    edge_node_indices = {}
    for edge_type in edge_types:
        if edge_type[0] == node_type and edge_type[1] == node_type:
            mask1 = torch.isin( graph[edge_type].edge_index[0], torch.arange(0,  graph[node_type].x.shape[0]).to(device)[node_indices])
            mask2 = torch.isin( graph[edge_type].edge_index[1], torch.arange(0,  graph[node_type].x.shape[0]).to(device)[node_indices])
            edge_index = (mask1) & (mask2)

        if edge_type[0] == node_type:
            edge_index =  torch.isin( graph[edge_type].edge_index[0], torch.arange(0,  graph[node_type].x.shape[0]).to(device)[node_indices])

        else:
            edge_index =  torch.isin( graph[edge_type].edge_index[1], torch.arange(0,  graph[node_type].x.shape[0]).to(device)[node_indices])


        graph[edge_type].edge_index = graph[edge_type].edge_index[:,edge_index]
        graph[edge_type].edges =  graph[edge_type].edges[edge_index, :]
        graph[edge_type].y =  graph[edge_type].y[edge_index]
        edge_node_indices[edge_type] = edge_index
    return edge_node_indices

class NeutralsHeteroGraphNetwork(AbstractModule):

    def __init__(self,
                 node_types, edge_types, edge_model, node_model,
                 global_model=None, use_globals=True, hidden_size=8, device="cuda",
                 use_edge_weights=True, use_node_weights=True, weight_mlp_layers=4, weight_mlp_channels=128,
                 weighted_mp = False, norm="batch_norm"):
        super(NeutralsHeteroGraphNetwork, self).__init__()
        self._use_globals = use_globals
        self.edge_types = edge_types
        self.node_types = node_types
        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.device = device
        # self.k =4000
        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001
        # self.select = SelectTopK(1 ,self.k_edges)
        # self.select_nodes = SelectTopK(1 ,self.k_nodes)

        with self._enter_variable_scope():
            self._edge_block = HeteroEdgeBlock(edge_types, edge_model_fn=edge_model)
            self._node_block = HeteroNodeBlock(node_types, edge_types, node_model_fn=node_model, weighted_mp = weighted_mp)
            if self._use_globals:
                # try use_nodes = False for node pruning test
                self._global_block = HeteroGlobalBlock(node_types, edge_types, global_model_fn=global_model, weighted_mp = weighted_mp)
        self._node_mlps = {}
        self._edge_mlps = {}
        # for node_type in node_types:
        #      self._node_mlps[node_type] = weight_mlp(1, hidden_channels=weight_mlp_channels, num_layers=weight_mlp_layers)()

        for edge_type in edge_types:
            self._edge_mlps[edge_type] = weight_mlp(1, hidden_channels=weight_mlp_channels,
                                                    num_layers=weight_mlp_layers,
                                                    norm=norm)()
        # self._node_mlp = weight_mlp(1)()
        for node_type in self.node_types:
            self._node_mlps[node_type] = weight_mlp(
                1,
                hidden_channels=weight_mlp_channels,
                num_layers=weight_mlp_layers,
                norm=norm
            )()
        #self._node_mlps['pvs'] = ones(device)
        # self._edge_mlp = weight_mlp(1)()
        # self._edge_mlps[('tracks','to','tracks')] = self._edge_mlp
        # self._edge_mlps[('tracks','to','PVs')] = ones(device)
        self._edge_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._edge_mlps.items()})
        self._node_models_model_dict = torch.nn.ModuleDict({str(i): j for i, j in self._node_mlps.items()})

        self._sigmoid = Sigmoid()
        self._use_edge_weights = use_edge_weights
        self._use_node_weights = use_node_weights
        self.edge_weights = {}
        self.node_weights = {}
        self.edge_logits = {}
        self.node_logits = {}
        self.edge_indices = {}
        self.node_indices = {}
        self.edge_node_pruning_indices = {}

    def forward(self, graph):
        node_input = self._edge_block(graph)

        for edge_type in self.edge_types:
            if self._use_edge_weights:
                graph_batch = node_input[edge_type[0]].batch[ node_input[edge_type].edge_index[0] ]
                self.edge_logits[edge_type] = self._edge_mlps[edge_type](node_input[edge_type].edges, graph_batch)
                self.edge_weights[edge_type] = self._sigmoid(self.edge_logits[edge_type])
            else:
                self.edge_weights[edge_type] = torch.ones((graph[edge_type].edges.shape[0], 1)).to(self.device)

        if self.edge_prune:
            for edge_type in self.edge_types:
                edge_indices = self.edge_weights[edge_type] > self.edge_weight_cut
                edge_indices = torch.arange(0, node_input[edge_type].edges.shape[0]).to(self.device)[
                    edge_indices.flatten()]
                self.edge_indices[edge_type] = edge_indices
                self.edge_weights[edge_type] = self.edge_weights[edge_type][edge_indices, :]
                edge_pruning(edge_indices, node_input, edge_type)

        global_input = self._node_block(node_input, self.edge_weights)
        # if self._use_node_weights:
        #     self.node_weights = self._sigmoid(self._edge_mlp(global_input['tracks'].x) )
        for node_type in self.node_types:
            if self._use_node_weights:
                self.node_logits[node_type] = self._node_mlps[node_type](global_input[node_type].x, global_input[node_type].batch)
                self.node_weights[node_type] = self._sigmoid(self.node_logits[node_type])
            else:
                self.node_weights[node_type] = torch.ones((graph[node_type].x.shape[0], 1)).to(self.device)

        if self.node_prune:
            for node_type in self.node_types:
                node_indices = self.node_weights[node_type] > self.node_weight_cut
                node_indices = torch.arange(0, graph[node_type].x.shape[0]).to(self.device)[node_indices.flatten()]
                self.node_indices[node_type] = node_indices
                edge_index = node_pruning(node_indices, global_input, node_type,
                                          [('chargedtree', 'to', 'neutrals')],
                                          device = self.device)
                self.edge_node_pruning_indices[node_type] = edge_index
                for key in edge_index.keys():
                    self.edge_weights[key] = self.edge_weights[key][edge_index[key]]

        # self.node_weights["tracks"] = self._sigmoid(self._node_mlps["tracks"](global_input["tracks"].x))
        # self.node_weights["PVs"] = self._node_mlps["PVs"](global_input["PVs"].x)
        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)
        else:
            return global_input