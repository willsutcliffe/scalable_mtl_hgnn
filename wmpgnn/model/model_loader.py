from wmpgnn.model.gnn_model import GNN
from wmpgnn.model.hetero_gnn_model import HeteroGNN
from wmpgnn.model.neutrals_hetero_gnn_model import NeutralsHeteroGNN


class ModelLoader:
    """ Class to set up the model """

    def __init__(self, config):
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
        elif model_type == "neutral_heterognn":
            nodes = config_loader.get("model")['node_types']
            edges = config_loader.get("model")['edge_types']
            edges = [(edge.split('_')[0], 'to', edge.split('_')[1]) for edge in edges]
            self.model = NeutralsHeteroGNN(node_types=nodes, edge_types=edges,
                                   mlp_output_size=config_loader.get("model.mlp_output_size"), 
                                   edge_op=config_loader.get("model.neutrals_classes"),
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
        elif model_type == "transformer":
            pass

    def get_model(self):
        return self.model