import sys
import os
import glob
from optparse import OptionParser

from itertools import chain

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
    parser.add_option("", "--bs", type=int, default=12,
                      dest="BS", help="batch size")
    parser.add_option("", "--gacc", type=int, default=1,
                      dest="GACC", help="gradient accumulation")
    parser.add_option('--cw', action='store_true', dest="CW", help='Calculate weights')
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # Load config file
    config_loader = ConfigLoader(option.CONFIG, environment_prefix="DL")

    # Load model
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()
    print(model)
    # Get the dataset glob it and load
    trn_paths = sorted(glob.glob(f'{option.INDIR}/training_data_*'))
    trn_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in trn_paths))
    val_paths = sorted(glob.glob(f'{option.INDIR}/validation_data_*'))
    val_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in val_paths))
    print("data read in")
    print(f"Train dataset: {len(trn_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
    # here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=4, num_workers=8, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=4, num_workers=8, drop_last=True) 

    # here we can do two stuff loading or during training
    if option.CW:  # CW = calculate pos weights
        pos_weight = get_hetero_weight(trn_loader)
    else:
        pos_weight = {'t_nodes': torch.tensor(11.3371),'tt_edges': torch.tensor(480.6347), 'LCA': torch.tensor([2.5026e-01, 4.8915e+02, 7.4080e+02, 7.5415e+03]) }
        # {'t_nodes': tensor(12.3371), 'tt_edges': tensor(480.1262), 'LCA': tensor([2.5026e-01, 9.8915e+02, 3.4080e+02, 4.5415e+03])}
    print(pos_weight)
    
    training(model, pos_weight, 5, 1, trn_loader, val_loader, accumulate_grad_batches=3) # need to add gradient accumulation
