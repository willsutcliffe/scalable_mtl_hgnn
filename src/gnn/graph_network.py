
from torch.nn import Linear, Sigmoid
from torch_geometric.nn import knn
from torch_geometric.nn.pool.select import SelectTopK

def make_weight_mlp(output_size):
    return lambda: MLP(in_channels=-1, hidden_channels=16,
                       out_channels=output_size, num_layers=4 , norm=None)

class GraphNetwork(AbstractModule):

    def __init__(self, edge_model, node_model, use_globals, global_model=None, hidden_size=8):
        #                  edge_block_opt=None, node_block_opt=None, global_block_opt=None):
        super(GraphNetwork, self).__init__()
        self.prune = False
        self.node_prune = False
        # self.k =4000
        self.k_edges = 1500
        self.k_nodes = 60
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
        # self.edge_weights = torch.ones_like(node_input.edges).cuda()
        # print(self.edge_weights.shape)
        if self.prune:
            # indices = torch.topk(self.edge_weights, self.k, dim=0).indices.squeeze(1)
            out = self.select(self.edge_weights, node_input.edgepos)
            indices = out.node_index
            self.indices = indices
            # print('initial edge shape ', node_input.edges.shape)
            updated_edges = node_input.edges[indices, :]
            updated_senders = node_input.senders[indices]
            updated_receivers = node_input.receivers[indices]
            updated_edge_pos = node_input.edgepos[indices]
            updated_y = node_input.y[indices]
            self.edge_weights = self.edge_weights[indices ,:]
            node_input.update( { 'edges': updated_edges,
                                 'senders' : updated_senders,
                                 'receivers' : updated_receivers,
                                 'edgepos' :  updated_edge_pos,
                                 'y' : updated_y} )
            # graph.update( { 'edges': updated_edges,
            #              'senders' : updated_senders,
            #             'receivers' : updated_receivers,
            #              'edgepos' :  updated_edge_pos,
            #              'y' : updated_y} )

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
        global_input = self._node_block(node_input, self.edge_weights)


        self.node_weights = self.sigmoid(self.node_mlp(global_input.nodes))
        if self.node_prune:
            # print("here")
            out = self.select_nodes(self.node_weights, global_input.batch)
            node_indices = out.node_index
            self.node_indices = node_indices
            self.node_weights = self.node_weights[node_indices]
            updated_nodes = global_input.nodes[node_indices, :]
            print(updated_nodes.shape)
            updated_batch = global_input.batch[node_indices]
            b1 = torch.isin(global_input.senders, torch.arange(0 ,global_input.nodes.shape[0]).cuda()[node_indices])
            b2 = torch.isin(global_input.receivers, torch.arange(0 ,global_input.nodes.shape[0]).cuda()[node_indices])
            edge_index = (b1) & (b2)
            self.node_edge_indices = edge_index
            self.edge_weights = self.edge_weights[edge_index]
            updated_edges = global_input.edges[edge_index, :]
            updated_senders = node_input.senders[edge_index]
            updated_receivers = node_input.receivers[edge_index]
            updated_edge_pos = node_input.edgepos[edge_index]
            updated_y = node_input.y[edge_index]
            global_input.update( {
                'nodes' : updated_nodes,
                'batch' : updated_batch,
                'edges': updated_edges,
                'senders' : updated_senders,
                'receivers' : updated_receivers,
                'edgepos' :  updated_edge_pos,
                'y' : updated_y} )
            graph.update( {
                'nodes' : updated_nodes,
                'batch' : updated_batch,
                'edges': updated_edges,
                'senders' : updated_senders,
                'receivers' : updated_receivers,
                'edgepos' :  updated_edge_pos,
                'y' : updated_y} )
            # if self.prune:
        # self.node_weights = torch.ones_like(node_input.nodes).cuda()

        # node_input.nodes = node_input.nodes*self.node_weights

        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)

        else:
            return global_input