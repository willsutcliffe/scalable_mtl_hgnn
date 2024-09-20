from prompt_toolkit.key_binding.bindings.named_commands import emacs_editing_mode
from torch.ao.quantization.backend_config.fbgemm import fbgemm_default_dynamic_float16_dtype_config

from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.edge_block import EdgeBlock
from wmpgnn.blocks.global_block import GlobalBlock
from wmpgnn.blocks.node_block import NodeBlock
import torch
from torch_geometric.nn.models import MLP
from torch.nn import Linear, Sigmoid
from torch_geometric.nn import knn
from torch_geometric.nn.pool.select import SelectTopK
import contextlib

def make_weight_mlp(output_size):
    return lambda: MLP(in_channels=-1, hidden_channels=16,
                       out_channels=output_size, num_layers=4 , norm=None)

def edge_pruning(edge_indices, graph):
    updated_edges = graph.edges[edge_indices, :]
    updated_senders = graph.senders[edge_indices]
    updated_receivers = graph.receivers[edge_indices]
    updated_edge_pos = graph.edgepos[edge_indices]
    updated_y = graph.y[edge_indices]

    graph.update({'edges': updated_edges,
                    'senders': updated_senders,
                    'receivers': updated_receivers,
                    'edgepos': updated_edge_pos,
                    'y': updated_y})


def node_pruning(node_indices, graph, device = "cuda"):
    updated_nodes = graph.nodes[node_indices, :]
    updated_batch =  graph.batch[node_indices]

    b1 = torch.isin( graph.senders, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    b2 = torch.isin( graph.receivers, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    edge_index = (b1) & (b2)
    updated_edges =  graph.edges[edge_index, :]
    # need to relabel senders and receivers to reflect the pruned away nodes
    print("How nodes should be ", graph.nodes[graph.receivers[edge_index]])
    edge_indices_concat = torch.concatenate([graph.receivers[edge_index], graph.senders[edge_index]])
    unique_elements, inverse_indices = torch.unique(edge_indices_concat, sorted=True, return_inverse=True)
    relabelled_tensor = inverse_indices
    updated_receivers = relabelled_tensor[:int(relabelled_tensor.shape[0] / 2)]
    updated_senders = relabelled_tensor[int(relabelled_tensor.shape[0] / 2):]
    updated_edge_pos =  graph.edgepos[edge_index]
    print("How nodes are ", updated_nodes[updated_receivers])
    updated_y =  graph.y[edge_index]
    graph.update({
        'nodes': updated_nodes,
        'batch': updated_batch,
        'edges': updated_edges,
        'senders': updated_senders,
        'receivers': updated_receivers,
        'edgepos': updated_edge_pos,
        'y': updated_y})
    return edge_index

def node_pruning2(node_indices, graph, device = "cuda"):
    b1 = torch.isin( graph.senders, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    b2 = torch.isin( graph.receivers, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    edge_index = (b1) & (b2)
    updated_edges =  graph.edges[edge_index, :]
    updated_senders = graph.senders[edge_index]
    updated_receivers = graph.receivers[edge_index]
    updated_edge_pos = graph.edgepos[edge_index]
    updated_y =  graph.y[edge_index]
    graph.update({
        'edges': updated_edges,
        'senders': updated_senders,
        'receivers': updated_receivers,
        'edgepos': updated_edge_pos,
        'y': updated_y})
    return edge_index

class GraphNetwork(AbstractModule):

    def __init__(self, edge_model, node_model, use_globals, global_model=None, hidden_size=8, device = "cuda"):
        #                  edge_block_opt=None, node_block_opt=None, global_block_opt=None):
        super(GraphNetwork, self).__init__()
        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.device = device
        # self.k =4000
        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001
        self._use_globals = use_globals
        with self._enter_variable_scope():
            self._edge_block = EdgeBlock(edge_model_fn=edge_model)
            self._node_block = NodeBlock(node_model_fn=node_model)
            if self._use_globals:
                self._global_block = GlobalBlock(global_model_fn=global_model)

        # add new weight matrices
        self.edge_linear = Linear(hidden_size ,1)
        self.edge_mlp = make_weight_mlp(1)()
        self.node_linear = Linear(hidden_size ,1)
        self.node_mlp = make_weight_mlp(1)()
        self.sigmoid = Sigmoid()
        self.select = SelectTopK(1 ,self.k_edges)
        self.select_nodes = SelectTopK(1 ,self.k_nodes)


    def forward(self, graph):

        node_input = self._edge_block(graph)

        self.edge_weights = self.sigmoid(self.edge_mlp(node_input.edges))
        if self.edge_prune:
            if self.prune_by_cut:
                edge_indices = self.edge_weights > self.edge_weight_cut
                edge_indices = torch.arange(0, graph.edges.shape[0]).to(self.device)[edge_indices.flatten()]

            else:
                out = self.select(self.edge_weights, node_input.receivers)
                edge_indices = out.node_index
            self.edge_indices = edge_indices
            self.edge_weights = self.edge_weights[edge_indices, :]
            edge_pruning(edge_indices, node_input)

        global_input = self._node_block(node_input, self.edge_weights)



        self.node_weights = self.sigmoid(self.node_mlp(global_input.nodes))
        if self.node_prune:
            if self.prune_by_cut:
                node_indices = self.node_weights > self.node_weight_cut
                node_indices = torch.arange(0, graph.nodes.shape[0]).to(self.device)[node_indices.flatten()]
            else:
                out = self.select_nodes(self.node_weights, global_input.batch)
                node_indices = out.node_index

            self.node_indices = node_indices
            self.node_weights = self.node_weights[node_indices]
            edge_index = node_pruning(node_indices, global_input, device = self.device)
            self.edge_node_pruning_indices = edge_index
            self.edge_weights = self.edge_weights[edge_index]


        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)

        else:
            return global_input

        # if self.prune:
        #     #indices = torch.topk(self.edge_weights, self.k, dim=0).indices.squeeze(1)
        #     y = torch.ones_like(torch.arange(32,dtype=torch.float)).cuda()
        #     x = self.edge_weights
        #     batch_y = torch.arange(32).cuda()
        #     batch_x = node_input.edgepos
        #     indices = knn(x, y, self.k, batch_x, batch_y)[1]
        #     self.indices = indices
        #     #print('initial edge shape ', node_input.edges.shape)
        #     updated_edges = node_input.edges[indices, :]
        #     updated_senders = node_input.senders[indices]
        #     updated_receivers = node_input.receivers[indices]
        #     updated_edge_pos = node_input.edgepos[indices]
        #     updated_y = node_input.y[indices]
        #     self.edge_weights = self.edge_weights[indices,:]

        # print(indices.shape)

        # print('after edge shape ', graph.edges.shape)
        # print("receiver shape ", graph.receivers.shape)
        # print("sender shape ", graph.senders.shape)
        # node_input.edges = node_input.edges*self.edge_weights


            # graph.update( {
            #     'nodes' : updated_nodes,
            #     'batch' : updated_batch,
            #     'edges': updated_edges,
            #     'senders' : updated_senders,
            #     'receivers' : updated_receivers,
            #     'edgepos' :  updated_edge_pos,
            #     'y' : updated_y} )
            # if self.prune:
        # self.node_weights = torch.ones_like(node_input.nodes).cuda()

        # node_input.nodes = node_input.nodes*self.node_weights