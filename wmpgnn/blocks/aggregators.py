from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add
from torch_scatter import scatter_mean
from torch_scatter import scatter_max
from torch_scatter import scatter_std

def globals_to_nodes(graph):
    return graph.graph_globals[graph.batch]

def receiver_nodes_to_edges(graph):
    # Later, add with tf.name_scope(name): ??
    # check if it works as tf.gather(graph.nodes, graph.senders)
    return graph.nodes[graph.receivers, :]

def sender_nodes_to_edges(graph):
    # Later, add with tf.name_scope(name): ??
    # check if it works as tf.gather(graph.nodes, graph.senders)
    return graph.nodes[graph.senders, :]

def globals_to_edges(graph):
    # _validate_broadcasted_graph(graph, GLOBALS, N_EDGE)
    #with tf.name_scope(name):
    #return utils_tf.repeat(graph.globals, graph.n_edge, axis=0,
    #                      sum_repeats_hint=num_edges_hint)
    return graph.graph_globals[graph.edgepos] #by Will
#     return graph.graph_globals.repeat(graph.num_edges, 1)

class EdgesToNodesAggregator(AbstractModule):
    # """Agregates sent or received edges into the corresponding nodes."""
    def __init__(self, use_sent_edges=False, b_edges=True, weighted=True, scatter_func = scatter_add):
        super(EdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        if graph.nodes is not None and graph.nodes.size()[0] is not None:
            num_nodes = graph.nodes.size()[0]
        else:
            num_nodes = graph.num_nodes

        indices = graph.senders if self._use_sent_edges else graph.receivers
        out = graph.edges.new_zeros(num_nodes, graph.edges.shape[1])
        #return scatter_add(graph.edges, indices, out=out, dim=0)
        # print('edge shape ', graph.edges.shape)
        # print('edge weights ', edge_weights.shape)
        # print(indices.shape)

        if not self._weighted:
            return self._scatter_func(graph.edges, indices, out=out, dim=0)
        elif self._b_edges == True:
            return self._scatter_func(graph.edges * edge_weights, indices, out=out, dim=0)
        else:
            return self._scatter_func(graph.edges * (1 - edge_weights), indices, out=out, dim=0)
        #return scatter_add(graph.edges, indices, out=out, dim=0)
        #return scatter_mean(graph.edges * edge_weights, indices, out=out, dim=0)


# reducer by adding edge features for corresponding nodes
# len(out)=No. of nodes
# def edge_to_node_reducer(graph, senders=True):
#     out = graph.edges.new_zeros(graph.num_nodes, graph.edges.shape[1])
#     if senders:
#         out = scatter_add(graph.edges, graph.senders, out=out, dim=0)
#     else:
#         out = scatter_add(graph.edges, graph.receivers, out=out, dim=0)
#     return out

class EdgesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None, weighted = True, b_edges=True, scatter_func = scatter_add):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1
        self._b_edges = b_edges
        self._weighted = weighted
        self._scatter_func = scatter_func

    def forward(self, graph, edge_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.edges, 0)
        else:
            out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.edges.shape[1])
            #out = graph.edges.new_zeros(len(graph.graph_globals), graph.edges.shape[1])
            #graph_index = torch.range(0, ((num_graphs)-1), dtype=torch.int64)
            #indices = graph_index.repeat(graph.num_edges, 1)
            if not self._weighted:
                out = self._scatter_func(graph.edges, graph.edgepos,out=out, dim=0)
            elif self._b_edges:
                out = self._scatter_func(graph.edges*edge_weights, graph.edgepos, out=out, dim=0)
            else:
                out = self._scatter_func(graph.edges * (1-edge_weights), graph.edgepos, out=out, dim=0)

            #out = scatter_mean(graph.edges * edge_weights, graph.edgepos, dim=0)
            #print("Edge -> global ", out.shape)
        return out


class NodesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None, weighted=True, b_nodes=True, scatter_func = scatter_add):
        super(NodesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1
        self._b_nodes = b_nodes
        self._weighted = weighted
        self._scatter_func = scatter_func
    def forward(self, graph, node_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.nodes, 0)
        else:
            out = graph.nodes.new_zeros(graph.graph_globals.shape[0], graph.nodes.shape[1])
            #graph_index = torch.range(0, ((num_graphs)-1), dtype=torch.int64)
            #indices = graph_index.repeat(graph.num_nodes, 1)
            #out = scatter_add(graph.nodes, graph.batch, dim=0)
            if not self._weighted:
                out = self._scatter_func(graph.nodes, graph.batch, out=out, dim=0)
            elif self._b_nodes:
                out = self._scatter_func(graph.nodes*node_weights, graph.batch, out=out, dim=0)
            else:
                out = self._scatter_func(graph.nodes * (1-node_weights), graph.batch, out=out, dim=0)
            #out = scatter_mean(graph.nodes * node_weights, graph.batch, dim=0)
            #print("Node -> global ", out.shape)
        return out



