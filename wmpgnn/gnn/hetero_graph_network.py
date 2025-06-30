from wmpgnn.blocks.abstract_module import AbstractModule
from wmpgnn.blocks.hetero_edge_block  import HeteroEdgeBlock
from wmpgnn.blocks.hetero_global_block import HeteroGlobalBlock
from wmpgnn.blocks.hetero_node_block import HeteroNodeBlock
import torch
import torch.nn as nn
from torch_geometric.nn.models import MLP
from torch.nn import Sigmoid


def weight_mlp(output_size, hidden_channels=16, num_layers=4, norm="batch_norm", drop_out=0.001):
    """
    Returns a function that creates a multilayer perceptron (MLP) for weight prediction.

    Args:
        output_size (int): Number of output channels for the MLP.
        hidden_channels (int, optional): Number of hidden units in each layer. Defaults to 16.
        num_layers (int, optional): Total number of layers in the MLP. Defaults to 4.
        norm (str, optional): Normalization method to use ("batch_norm", "layer_norm", etc.). Defaults to "batch_norm".

    Returns:
        Callable: A lambda function that creates an MLP instance when called.
    """
    return lambda: MLP(in_channels=-1, hidden_channels=hidden_channels,
                       out_channels=output_size, num_layers=num_layers, norm=norm, dropout=drop_out)


def ones(device):
    """
    Returns a function that outputs a tensor of ones with shape (N, 1), where N is the number of input rows.

    Args:
        device (str): The device ("cpu" or "cuda") to place the tensor on.

    Returns:
        Callable: A lambda function that takes a tensor `x` and returns a tensor of ones with shape (x.shape[0], 1).
    """
    return lambda x: torch.ones((x.shape[0], 1)).to(device)


def edge_pruning(edge_indices, graph, edge_type):
    """
    Prunes edges of the given type from a heterogeneous graph based on the provided indices.

    Args:
        edge_indices (Tensor): Indices of edges to keep.
        graph (HeteroData): The heterogeneous graph data object.
        edge_type (tuple): The edge type tuple (src_type, relation_type, dst_type).
    """
    graph[edge_type].edges = graph[edge_type].edges[edge_indices]
    graph[edge_type].edge_index = torch.vstack(
        [graph[edge_type].edge_index[0][edge_indices],
         graph[edge_type].edge_index[1][edge_indices]])
    graph[edge_type].y = graph[edge_type].y[edge_indices]


def node_pruning(node_indices, graph, node_type, edge_types, device = "cuda"):
    """
    Prunes edges associated with pruned nodes from a heterogeneous graph based on the given indices.
    Nodes are not explicitly removed as this was found to reduce performance.

    Args:
        node_indices (Tensor): Indices of nodes to retain.
        graph (HeteroData): The heterogeneous graph data object.
        node_type (str): The node type to prune.
        edge_types (list): List of edge types to consider for pruning.
        device (str, optional): Device on which to perform the computation. Defaults to "cuda".

    Returns:
        dict: A dictionary mapping edge types to the pruned edge index masks.
    """
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


def faster_node_pruning(node_indices, graph, node_type, edge_types, device="cuda"):
    # Precompute a boolean mask for valid nodes.
    num_nodes = graph[node_type].x.shape[0]
    valid_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    valid_mask[node_indices] = True

    edge_node_indices = {}
    for edge_type in edge_types:
        if edge_type[0] == node_type and edge_type[1] == node_type:
            # Use the valid mask directly for both source and target.
            mask = valid_mask[graph[edge_type].edge_index[0]] & valid_mask[graph[edge_type].edge_index[1]]
        elif edge_type[0] == node_type:
            mask = valid_mask[graph[edge_type].edge_index[0]]
        else:
            mask = valid_mask[graph[edge_type].edge_index[1]]

        graph[edge_type].edge_index = graph[edge_type].edge_index[:, mask]
        graph[edge_type].edges = graph[edge_type].edges[mask, :]
        graph[edge_type].y = graph[edge_type].y[mask]
        edge_node_indices[edge_type] = mask
    return edge_node_indices

class HeteroGraphNetwork(AbstractModule):
    """
    Heterogeneous graph neural network that performs edge, node, and optional global updates,
    with per-type learned weighting and pruning.

    The update pipeline per forward pass:
      1. **Edge Update**
         Uses `HeteroEdgeBlock` to update each edge type’s features.
      2. **Edge Weighting**
         For each edge type, applies a small MLP + Sigmoid to produce importance weights.
      3. **Edge Pruning (optional)**
         Keeps only edges above a threshold or the top‑K per type.
      4. **Node Update**
         Uses `HeteroNodeBlock` to update each node type’s features, conditioned on pruned edge weights.
      5. **Node Weighting**
         For each node type, applies a small MLP + Sigmoid to produce importance weights.
      6. **Node Pruning (optional)**
         Keeps only nodes above a threshold or the top‑K for designated types, pruning incident edges.
      7. **Global Update (optional)**
         Uses `HeteroGlobalBlock` to update the shared global features from all edge and node aggregates.

    Attributes:
        edge_types (List[tuple]): Relation keys `(src_type, dst_type, rel_key)`.
        node_types (List[str]): Keys for each node set.
        _edge_block (HeteroEdgeBlock): Module to update heterogeneous edges.
        _node_block (HeteroNodeBlock): Module to update heterogeneous nodes.
        _global_block (HeteroGlobalBlock): Module to update global features.
        _edge_mlps (Dict[edge_type, MLP]): MLPs predicting scalar weight per edge.
        _node_mlps (Dict[node_type, MLP]): MLPs predicting scalar weight per node.
        _sigmoid (Sigmoid): Sigmoid activation used on logits to produce weights.
        use_edge_weights (bool): Whether to learn and apply edge weights.
        use_node_weights (bool): Whether to learn and apply node weights.
        edge_prune (bool): Enable edge pruning step.
        node_prune (bool): Enable node pruning step.
        prune_by_cut (bool): If True, prune by threshold; otherwise by top‑K.
        k_edges (int): Top‑K edges to keep when pruning by rank.
        k_nodes (int): Top‑K nodes to keep when pruning by rank.
        edge_weight_cut (float): Threshold for edge pruning.
        node_weight_cut (float): Threshold for node pruning.
        device (str): Device string for tensor creation.
        edge_weights (Dict): Stored edge‑type weight tensors per forward pass.
        node_weights (Dict): Stored node‑type weight tensors per forward pass.
        edge_indices (Dict): Indices of kept edges after pruning.
        node_indices (Dict): Indices of kept nodes after pruning.
        edge_node_pruning_indices (Dict): Masks of surviving edges after node pruning.
    """
    def __init__(self,
                 node_types, edge_types, edge_model, node_model,
                 global_model=None, use_globals=True, hidden_size=8, device="cuda",
                 use_edge_weights=True, use_node_weights=True, weight_mlp_layers=4, weight_mlp_channels=128,
                 weighted_mp = False, norm="batch_norm", drop_out=0., nFT_layers=False):
        """
        Initialize the HeteroGraphNetwork.

        Args:
            node_types (List[str]): Node‑set keys in the heterogeneous graph.
            edge_types (List[tuple]): Relation keys `(src, dst, rel_key)` for edges.
            edge_model (callable): Zero‑argument factory returning edge update modules.
            node_model (callable): Zero‑argument factory returning node update modules.
            global_model (callable, optional): Zero‑argument factory for global update module.
            use_globals (bool): If True, include global update block.
            hidden_size (int): Unused; kept for interface consistency.
            device (str): Device identifier for creating default weight tensors.
            use_edge_weights (bool): Learn and apply per‑edge weights.
            use_node_weights (bool): Learn and apply per‑node weights.
            weight_mlp_layers (int): Number of layers in each weighting MLP.
            weight_mlp_channels (int): Hidden size of each weighting MLP.
            weighted_mp (bool): Pass weights into message‑passing aggregators.
            norm (str): Normalization type ("batch_norm", etc.) for weighting MLPs.
        """
        super(HeteroGraphNetwork, self).__init__()

        self._use_globals = use_globals
        self.edge_types = edge_types
        self.node_types = node_types
        self.FT = nFT_layers
        self.edge_prune = False
        self.node_prune = False
        self.prune_by_cut = False
        self.device = device

        self.k_edges = 20
        self.k_nodes = 70
        self.edge_weight_cut = 0.001
        self.node_weight_cut = 0.001


        with self._enter_variable_scope():
            self._edge_block = HeteroEdgeBlock(edge_types, edge_model_fn=edge_model)
            self._node_block = HeteroNodeBlock(node_types, edge_types, node_model_fn=node_model, weighted_mp = weighted_mp)
            if self._use_globals:

                self._global_block = HeteroGlobalBlock(node_types, edge_types, global_model_fn=global_model, weighted_mp = weighted_mp)
        self._node_mlps = {}
        self._edge_mlps = {}

        for edge_type in edge_types:
            self._edge_mlps[edge_type] = weight_mlp(1, hidden_channels=weight_mlp_channels,
                                                    num_layers=weight_mlp_layers,
                                                    norm=norm, drop_out=drop_out)()  # MLPS for edge classification after each block

        self._node_mlps['tracks'] = weight_mlp(1, hidden_channels=weight_mlp_channels,
                                               num_layers=weight_mlp_layers,
                                               norm=norm, drop_out=drop_out)()  # MLPs for node classification after each block
        if self.FT:
            self._node_mlps['frag'] = weight_mlp(1, hidden_channels=weight_mlp_channels,
                                                num_layers=weight_mlp_layers,
                                                norm=norm, drop_out=drop_out)()  # MLPs for fragmentation classification after each block
            self._node_mlps['ft'] = weight_mlp(3, hidden_channels=weight_mlp_channels,  # Maybe take a smaller one?
                                                num_layers=weight_mlp_layers,
                                                norm=norm, drop_out=drop_out)()  # MLPs for FT after each block

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

    def forward(self, graph, pid_nodes):
        """
        Execute one GNN pass over a heterogeneous graph.

        Args:
            graph (HeteroData or similar): A heterogeneous graph object where each
                key `edge_type` maps to a small Data object containing:
                - `edges`: current edge features
                - `edge_index`: source/destination indices
                - Node feature attributes on `graph[src_type].x` and `graph[dst_type].x`
                - A `globals` node set with `x` for global features.

        Returns:
            HeteroData: The same `graph` with updated `.edges`, `.x`, and
            optionally `.globals.x` according to the configured blocks.
        """
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
                if edge_type == ('tracks', 'to', 'tracks'):
                    mask = self.edge_weights[edge_type] > self.edge_weight_cut
                    edge_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.edge_indices[edge_type] = edge_indices
                    self.edge_weights[edge_type] = self.edge_weights[edge_type][edge_indices, :]
                    edge_pruning(edge_indices, node_input, edge_type)

        global_input = self._node_block(node_input, self.edge_weights)

        for node_type in self.node_types:
            if self._use_node_weights and node_type != "pvs":
                self.node_logits[node_type] = self._node_mlps[node_type](global_input[node_type].x, global_input[node_type].batch)
                self.node_weights[node_type] = self._sigmoid(self.node_logits[node_type])
            else:
                self.node_weights[node_type] = torch.ones((graph[node_type].x.shape[0], 1)).to(self.device)
        
        if self.FT:  # Additional Layers for fragmentation particle identification and FT
            # Fragmentation
            self.node_logits["frag"] = self._node_mlps["frag"](global_input["tracks"].x, global_input["tracks"].batch)
            self.node_weights["frag"] = self._sigmoid(self.node_logits["frag"])
            # FT
            combined_graph = torch.cat([global_input["tracks"].x, pid_nodes], dim=1)  # catting the pid values before FT inference
            combined_graph = torch.cat([combined_graph, self.node_weights['tracks']], dim=1)  # catting the node weights
            # I think its good to add here the node weights as well
            self.node_logits["ft"] = self._node_mlps["ft"](combined_graph, global_input["tracks"].batch)
            self.node_weights["ft"] = torch.softmax(self.node_logits["ft"])  # this should be softmax


        if self.node_prune:
            for node_type in self.node_types:
                if node_type == "tracks":
                    mask = self.node_weights[node_type] > self.node_weight_cut
                    node_indices = torch.nonzero(mask, as_tuple=True)[0]
                    self.node_indices[node_type] = node_indices
                    edge_index = faster_node_pruning(node_indices, global_input, node_type,
                                              [('tracks', 'to', 'tracks')],
                                              device = self.device)
                    self.edge_node_pruning_indices[node_type] = edge_index
                    for key in edge_index.keys():
                        self.edge_weights[key] = self.edge_weights[key][edge_index[key]]

        if self._use_globals:
            return self._global_block(global_input, self.edge_weights, self.node_weights)
        else:
            return global_input