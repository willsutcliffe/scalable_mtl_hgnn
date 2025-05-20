import sys
import os
import glob
from optparse import OptionParser

import torch
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.lightning.lightning_module import training
from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.util.functions import get_hetero_weight


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
    trn_paths = sorted(glob.glob(f'{option.INDIR}/training_data_*'))
    trn_dataset = []
    for path in trn_paths:
        trn_dataset.append(torch.load(path, weights_only=False))
    val_paths = sorted(glob.glob(f'{option.INDIR}/validation_data_*'))
    val_dataset = []
    for path in val_paths:
        val_paths.append(torch.load(path, weights_only=False))

    # here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=6, num_workers=8, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=6, num_workers=8, drop_last=True) 

    # here we can do two stuff loading or during training
    if option.CW:  # CW = calculate pos weights
        pos_weight = get_hetero_weight(trn_loader)
    else:
        pos_weight = {'t_nodes': torch.tensor(11.8125),'tt_edges': torch.tensor(573.6347), 'LCA': torch.tensor([2.5022e-01, 4.8136e+02, 7.8440e+02, 7.4500e+03]) }
    print(pos_weight)
    
    training(model, pos_weight, 1, 1, trn_loader, val_loader) # need to add gradient accumulation
