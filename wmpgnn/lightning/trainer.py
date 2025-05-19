import torch

from torch_geometric.loader import DataLoader

from optparse import OptionParser

from wmpgnn.lightning.lightning_module import training
from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.model.hetero_gnn_model import HeteroGNN
from wmpgnn.util.functions import get_hetero_weight


class ModelLoader:
    """ Class to set up the model """
    def __init__(self, config):
        config_loader = config
        model_type = config_loader.get("model.type")
        model_type="heterognn"                   
        if model_type == "heterognn":
            nodes = config_loader.get("model")['node_types']
            edges = config_loader.get("model")['edge_types']
            edges = [(edge.split('_')[0],'to', edge.split('_')[1]) for edge in edges]
            self.model = HeteroGNN(node_types = nodes, edge_types = edges,
                             mlp_output_size=config_loader.get("model.mlp_output_size"), edge_op=4,
                             num_blocks=config_loader.get("model.gnn_layers"),
                             mlp_layers = config_loader.get("model.mlp_layers"),
                             mlp_channels = config_loader.get("model.mlp_channels"),
                             weight_mlp_channels = config_loader.get("model.weight_mlp_channels"),
                             weight_mlp_layers= config_loader.get("model.weight_mlp_layers"),
                             use_edge_weights = config_loader.get("model.use_edge_weights"),
                             use_node_weights = config_loader.get("model.use_node_weights"),
                             weighted_mp = config_loader.get("model.weighted_mp")
                            )    
    def get_model(self):
        return self.model
    

if __name__ == "__main__":
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--indir", type=str, default=None,
                      dest="INDIR", help="Input directory where files are gobbled from")
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # Load config file
    config_loader = ConfigLoader(option.CONFIG, environment_prefix="DL")

    # Load model
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()

    # Get the dataset glob it and load
    # option.INDIR
    trn_dataset = torch.load("training_data_4000.pt", weights_only=False)
    val_dataset = torch.load("validation_data_1000.pt", weights_only=False)

    # here we can check what kind of gpu it is
    trn_loader = DataLoader(trn_dataset, batch_size=6, num_workers=15, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=6, num_workers=15, drop_last=True) 

    # here we can do two stuff loading or during training
    if option.CW:  # CW = calculate pos weights
        pos_weight = get_hetero_weight(trn_loader)
    else:
        pos_weight = {'t_nodes': torch.tensor(11.8125),'tt_edges': torch.tensor(573.6347), 'LCA': torch.tensor([2.5022e-01, 4.8136e+02, 7.8440e+02, 7.4500e+03]) }
    print(pos_weight)
    
    training(model, pos_weight, 10, 1, trn_loader, val_loader) # need to add gradient accumulation
