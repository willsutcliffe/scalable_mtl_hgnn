from wmpgnn.blocks.abstract_module import AbstractModule
import torch

from torch_scatter import scatter_add
from torch_scatter import scatter_mean

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
    def __init__(self, use_sent_edges=False):
        super(EdgesToNodesAggregator, self).__init__()
        self._use_sent_edges = use_sent_edges

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
        return scatter_add(graph.edges * edge_weights, indices, out=out, dim=0)
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
    def __init__(self, num_graphs=None):
        super(EdgesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1

    def forward(self, graph, edge_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.edges, 0)
        else:
            #out = graph.edges.new_zeros(len(graph.graph_globals), graph.edges.shape[1])
            #graph_index = torch.range(0, ((num_graphs)-1), dtype=torch.int64)
            #indices = graph_index.repeat(graph.num_edges, 1)
            #out = scatter_add(graph.edges, graph.edgepos, dim=0)
            out = scatter_add(graph.edges*edge_weights, graph.edgepos, dim=0)
            #out = scatter_mean(graph.edges * edge_weights, graph.edgepos, dim=0)
            #print("Edge -> global ", out.shape)
        return out


class NodesToGlobalsAggregator(AbstractModule):
    def __init__(self, num_graphs=None):
        super(NodesToGlobalsAggregator, self).__init__()
        self._num_graphs = num_graphs
        self._num_graphs = 1

    def forward(self, graph, node_weights):
        if self._num_graphs is None:
            out = torch.sum(graph.nodes, 0)
        else:
            #out = graph.nodes.new_zeros(len(graph.graph_globals), graph.nodes.shape[1])
            #graph_index = torch.range(0, ((num_graphs)-1), dtype=torch.int64)
            #indices = graph_index.repeat(graph.num_nodes, 1)
            #out = scatter_add(graph.nodes, graph.batch, dim=0)
            out = scatter_add(graph.nodes*node_weights, graph.batch, dim=0)
            #out = scatter_mean(graph.nodes * node_weights, graph.batch, dim=0)
            #print("Node -> global ", out.shape)
        return out
