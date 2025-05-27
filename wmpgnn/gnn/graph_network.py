from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.edge_block import EdgeBlock
from wmpgnn.blocks.global_block import GlobalBlock
from wmpgnn.blocks.node_block import NodeBlock
import torch
from torch_geometric.nn.models import MLP
from torch.nn import Linear, Sigmoid, Softmax
from torch_geometric.nn.pool.select import SelectTopK
import contextlib

def make_weight_mlp(output_size, hidden_channels=16, num_layers=4, norm="batch_norm"):
    """
    Factory for creating a small MLP that predicts scalar weights per element.

    Args:
        output_size (int): Dimension of the MLP output (typically 1 for weights).
        hidden_channels (int): Number of hidden units in each intermediate layer.
        num_layers (int): Total number of linear layers (including input and output).
        norm (str): Normalization type to use in the MLP (e.g. "batch_norm", "layer_norm").

    Returns:
        callable: A zero‐argument constructor for a torch_geometric.nn.models.MLP
                  with the specified configuration.
    """
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
                       out_channels=output_size, num_layers=num_layers, norm=norm)

def edge_pruning(edge_indices, graph):
    """
    Prune edges from a graph in-place by selecting a subset of indices.

    This removes all fields (`edges`, `senders`, `receivers`, `edgepos`, `y`)
    except those at the provided `edge_indices`.

    Args:
        edge_indices (LongTensor): Indices of edges to keep.
        graph (Data): A PyTorch Geometric Data object with attributes:
            - edges, senders, receivers, edgepos, y

    Side‑Effects:
        Mutates `graph` to only contain the selected edges.
    """
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


def node_pruning2(node_indices, graph, device = "cuda"):
    """
    Prune nodes (and implicitly edges) from a graph in-place.

    Keeps only nodes at `node_indices`, then removes any edges
    that reference discarded nodes.

    Args:
        node_indices (LongTensor): Indices of nodes to keep.
        graph (Data): A PyTorch Geometric Data object with attributes:
            - nodes, batch, edges, senders, receivers, edgepos, y
        device (str): Device on which tensors (e.g. graph.senders) reside.

    Returns:
        BoolTensor: A mask of length E indicating which edges remain.
    """
    updated_nodes = graph.nodes[node_indices, :]
    updated_batch =  graph.batch[node_indices]

    b1 = torch.isin( graph.senders, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    b2 = torch.isin( graph.receivers, torch.arange(0,  graph.nodes.shape[0]).to(device)[node_indices])
    edge_index = (b1) & (b2)
    updated_edges =  graph.edges[edge_index, :]
    edge_indices_concat = torch.concatenate([graph.receivers[edge_index], graph.senders[edge_index]])
    unique_elements, inverse_indices = torch.unique(edge_indices_concat, sorted=True, return_inverse=True)
    relabelled_tensor = inverse_indices
    updated_receivers = relabelled_tensor[:int(relabelled_tensor.shape[0] / 2)]
    updated_senders = relabelled_tensor[int(relabelled_tensor.shape[0] / 2):]
    updated_edge_pos =  graph.edgepos[edge_index]

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

def node_pruning(node_indices, graph, device = "cuda"):
    """
    Prunes only edges associated with nodes to be pruned from a graph in-place.

    Args:
        node_indices (LongTensor): Indices of nodes to keep.
        graph (Data): A PyTorch Geometric Data object with attributes:
            - nodes, batch, edges, senders, receivers, edgepos, y
        device (str): Device on which tensors (e.g. graph.senders) reside.

    Returns:
        BoolTensor: A mask of length E indicating which edges remain.
    """
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
    """
    End‐to‐end graph neural network combining edge, node, and global updates,
    with optional learned edge/node weighting and pruning.

    The computation proceeds:
      1. EdgeBlock: updates edge features.
      2. Weight MLP + Sigmoid: computes per‐edge importance weights.
      3. Optional edge pruning (by threshold or top‐K).
      4. NodeBlock: updates node features using (pruned) edge weights.
      5. Weight MLP + Sigmoid: computes per‐node importance weights.
      6. Optional node pruning (by threshold or top‐K).
      7. GlobalBlock: updates graph‐level features from edges, nodes, and globals.

    Attributes:
        edge_prune (bool): Whether to prune edges after weighting.
        node_prune (bool): Whether to prune nodes after weighting.
        prune_by_cut (bool): If True use thresholds; else use top‐K selection.
        k_edges (int): Number of edges to keep when using top‐K edge pruning.
        k_nodes (int): Number of nodes to keep when using top‐K node pruning.
        edge_weight_cut (float): Threshold for edge pruning by cut.
        node_weight_cut (float): Threshold for node pruning by cut.
        _edge_block (EdgeBlock): Module for edge updates.
        _node_block (NodeBlock): Module for node updates.
        _global_block (GlobalBlock): Module for global updates (if enabled).
        edge_mlp (MLP): MLP predicting a scalar weight per edge.
        node_mlp (MLP): MLP predicting a scalar weight per node.
        sigmoid (Sigmoid): Sigmoid activation for weights.
        softmax (Softmax): (Unused) softmax over weights.
        select (SelectTopK): Top‐K selector for edges.
        select_nodes (SelectTopK): Top‐K selector for nodes.
        use_edge_weights (bool): Whether to apply learned edge weights.
        use_node_weights (bool): Whether to apply learned node weights.
    """
    def __init__(self, edge_model, node_model, use_globals, global_model=None, hidden_size=8, device = "cuda",
         use_edge_weights = True, use_node_weights = True, weight_mlp_layers=4, weight_mlp_channels=128,
                 weighted_mp = False, norm = "batch_norm"):
        """
        Initialize the GraphNetwork.

        Args:
            edge_model (callable): Zero‐arg constructor for edge update nn.Module.
            node_model (callable): Zero‐arg constructor for node update nn.Module.
            use_globals (bool): Whether to include global updates.
            global_model (callable, optional): Zero‐arg constructor for global update nn.Module.
            hidden_size (int): Hidden dimension size (unused here).
            device (str): Device string for tensors.
            use_edge_weights (bool): If True, learn per‐edge weights.
            use_node_weights (bool): If True, learn per‐node weights.
            weight_mlp_layers (int): Number of layers in weight MLPs.
            weight_mlp_channels (int): Hidden channels in weight MLPs.
            weighted_mp (bool): If True, pass weights into message‐passing blocks.
            norm (str): Normalization type for MLPs.
        """
        super(GraphNetwork, self).__init__()

        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.device = device
        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001
        self._use_globals = use_globals
        with self._enter_variable_scope():
            self._edge_block = EdgeBlock(edge_model_fn=edge_model)
            self._node_block = NodeBlock(node_model_fn=node_model, weighted_mp=weighted_mp)
            if self._use_globals:
                self._global_block = GlobalBlock(global_model_fn=global_model, use_nodes=True, weighted_mp=weighted_mp)

        self.edge_mlp = make_weight_mlp(1, hidden_channels=weight_mlp_channels, num_layers=weight_mlp_layers, norm=norm)()

        self.node_mlp = make_weight_mlp(1, hidden_channels=weight_mlp_channels, num_layers=weight_mlp_layers, norm=norm)()
        self.sigmoid = Sigmoid()
        self.softmax = Softmax(dim=0)
        self.select = SelectTopK(1 ,self.k_edges)
        self.select_nodes = SelectTopK(1 ,self.k_nodes)
        self.use_edge_weights = use_edge_weights
        self.use_node_weights = use_node_weights


    def forward(self, graph):
        """
        Forward pass through edge, node, and optional global blocks, with
        learned weighting and pruning.

         Args:
            graph: A graph‐like object with attributes:
                - `edges`:      Tensor [E, D_e] of current edge features.
                - `senders`:    LongTensor [E] of sender node indices.
                - `receivers`:  LongTensor [E] of receiver node indices.
                - `graph_globals`: Tensor [G, D_g] of per‐graph globals.
                - `batch`:      LongTensor [N], graph indices for each node.
                - `edgepos`:    LongTensor [E] mapping each edge to a graph index.
                - `nodes`:      Tensor [N, D_n] of node features.

        Returns:
            Data: Updated Data object with new `edges`, `nodes`, and
                  optionally `graph_globals`.
        """
        node_input = self._edge_block(graph)

        self.edges = node_input.edges
        if self.use_edge_weights:
            self.edge_logits = self.edge_mlp(node_input.edges, node_input.edgepos)
            self.edge_weights = self.sigmoid(self.edge_logits)
        else:
            self.edge_weights = torch.ones( (node_input.edges.shape[0],1) ).to(self.device)

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
        if self.use_node_weights:
            self.node_logits = self.node_mlp(global_input.nodes, node_input.batch)
            self.node_weights = self.sigmoid(self.node_logits)
        else:
            self.node_weights = torch.ones( (global_input.nodes.shape[0],1) ).to(self.device)

        if self.node_prune:
            if self.prune_by_cut:
                node_indices = self.node_weights > self.node_weight_cut
                node_indices = torch.arange(0, graph.nodes.shape[0]).to(self.device)[node_indices.flatten()]
            else:
                out = self.select_nodes(self.node_weights, global_input.batch)
                node_indices = out.node_index

            self.node_indices = node_indices
            edge_index = node_pruning(node_indices, global_input, device = self.device)
            self.edge_node_pruning_indices = edge_index
            self.edge_weights = self.edge_weights[edge_index]

        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)
        else:
            return global_input

