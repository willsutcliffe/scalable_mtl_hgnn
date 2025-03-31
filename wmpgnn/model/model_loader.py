from wmpgnn.model.gnn_model import GNN
from wmpgnn.model.hetero_gnn_model import HeteroGNN


class ModelLoader:
    """
    A factory class for loading and initializing Graph Neural Network models.

    This class dynamically instantiates GNN models based on configuration parameters,
    supporting both standard Message Passing GNN and Heterogeneous GNN architectures.

    Attributes:
        model: The instantiated GNN model (either GNN or HeteroGNN instance).
    """
    def __init__(self, config):
        """
        Initialize the ModelLoader with configuration and create the appropriate GNN model.

        Args:
            config: Configuration object with a get() method for accessing parameters.
                   Must contain model type and all required model parameters.

        Configuration Parameters:
            model.type (str): Model type, either "mpgnn" or "heterognn"
            model.mlp_output_size (int): Output size of MLP layers
            model.gnn_layers (int): Number of GNN blocks/layers
            model.mlp_layers (int): Number of MLP layers
            model.mlp_channels (int): Number of channels in MLP layers
            model.weight_mlp_channels (int): Number of channels in weight MLP layers
            model.weight_mlp_layers (int): Number of weight MLP layers
            model.use_edge_weights (bool): Whether to use edge weights
            model.use_node_weights (bool): Whether to use node weights
            model.weighted_mp (bool): Whether to use weighted message passing
            model.norm (str): Normalization type

        Additional for heterognn:
            model.node_types (list): List of node type names
            model.edge_types (list): List of edge type strings in format "source_target"

        Raises:
            KeyError: If required configuration parameters are missing
            ValueError: If model type is not supported
        """
        config_loader = config
        model_type = config_loader.get("model.type")
        if model_type == "mpgnn":
            self.model = GNN(mlp_output_size=config_loader.get("model.mlp_output_size"), edge_op=config_loader.get("model.LCA_classes"),#,node_op=3,
                             num_blocks=config_loader.get("model.gnn_layers"),
                             mlp_layers=config_loader.get("model.mlp_layers"),
                             mlp_channels=config_loader.get("model.mlp_channels"),
                             weight_mlp_channels=config_loader.get("model.weight_mlp_channels"),
                             weight_mlp_layers=config_loader.get("model.weight_mlp_layers"),
                             use_edge_weights=config_loader.get("model.use_edge_weights"),
                             use_node_weights=config_loader.get("model.use_node_weights"),
                             weighted_mp=config_loader.get("model.weighted_mp"),
                             norm=config_loader.get("model.norm")
                             )
        elif model_type == "heterognn":
            nodes = config_loader.get("model")['node_types']
            edges = config_loader.get("model")['edge_types']
            edges = [(edge.split('_')[0], 'to', edge.split('_')[1]) for edge in edges]
            self.model = HeteroGNN(node_types=nodes, edge_types=edges,
                                   mlp_output_size=config_loader.get("model.mlp_output_size"), edge_op=config_loader.get("model.LCA_classes"),
                                   num_blocks=config_loader.get("model.gnn_layers"),
                                   mlp_layers=config_loader.get("model.mlp_layers"),
                                   mlp_channels=config_loader.get("model.mlp_channels"),
                                   weight_mlp_channels=config_loader.get("model.weight_mlp_channels"),
                                   weight_mlp_layers=config_loader.get("model.weight_mlp_layers"),
                                   use_edge_weights=config_loader.get("model.use_edge_weights"),
                                   use_node_weights=config_loader.get("model.use_node_weights"),
                                   weighted_mp=config_loader.get("model.weighted_mp"),
                                   norm=config_loader.get("model.norm")
                                   )


    def get_model(self):
        """
        Get the instantiated GNN model.

        Returns:
            GNN or HeteroGNN: The initialized model instance based on configuration.
                             Returns GNN for "mpgnn" type or HeteroGNN for "heterognn" type.
        """
        return self.model