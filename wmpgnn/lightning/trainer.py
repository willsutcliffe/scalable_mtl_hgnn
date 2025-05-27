import sys
import os
import glob
from optparse import OptionParser

from itertools import chain

import torch
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.lightning.lightning_module import training
from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.util.functions import get_hetero_weight

class LazyTorchDataset(Dataset):  # this doesnt seem to work that well...
    def __init__(self, file_pattern):
        self.files = sorted(glob.glob(file_pattern))
        self.file_lengths = []
        for f in self.files:
            data = torch.load(f, weights_only=False, map_location='cpu')
            self.file_lengths.append(len(data))
        print(self.file_lengths)
        self.cumulative_lengths = []
        total = 0
        for length in self.file_lengths:
            total += length
            self.cumulative_lengths.append(total)
        
        self._cache_file_idx = None
        self._cache_data = None

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        file_idx = 0
        while idx >= self.cumulative_lengths[file_idx]:
            file_idx += 1
        
        local_idx = idx if file_idx == 0 else idx - self.cumulative_lengths[file_idx - 1]
        
        # Load file if not cached
        if self._cache_file_idx != file_idx:
            self._cache_data = torch.load(self.files[file_idx], weights_only=False, map_location='cpu')
            self._cache_file_idx = file_idx
        
        return self._cache_data[local_idx]


if __name__ == "__main__":
    # python trainer.py --indir /eos/user/y/yukaiz/DFEI_data/inclusive/ --config /afs/cern.ch/work/y/yukaiz/public/weighted_MP_gnn/config_files/simple.yaml --bs 3 --gacc 4 --ngpu 4
    # python trainer.py --indir /eos/user/y/yukaiz/DFEI_data/inclusive/ --config  ../../config_files//simple.yaml --bs 12 --gacc 1 --ngpu 1 --cpt /eos/user/y/yukaiz/SWAN_projects/weighted_MP_gnn/wmpgnn/lightning/lightning_logs/version_2/checkpoints/epoch-epoch=07.ckpt
    # bash /afs/cern.ch/work/y/yukaiz/public/jobs/train_hgnn.sh /eos/user/y/yukaiz/DFEI_data/inclusive/ /afs/cern.ch/work/y/yukaiz/public/weighted_MP_gnn/config_files/simple.yaml 6 2 1
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--indir", type=str, default=None,
                      dest="INDIR", help="Input directory where files are gobbled from")
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    parser.add_option("", "--ngpu", type=int, default=1,
                      dest="NGPU", help="number of used gpu")
    parser.add_option("", "--bs", type=int, default=12,
                      dest="BS", help="batch size")
    parser.add_option("", "--gacc", type=int, default=1,
                      dest="GACC", help="gradient accumulation")
    parser.add_option("", "--cpt", type=str, default=None,
                      dest="CHECKPOINT", help="pass a checkpoint to continue training")
    parser.add_option('--cw', action='store_true', dest="CW", help='Calculate weights')
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # Load config file
    config_loader = ConfigLoader(option.CONFIG, environment_prefix="DL")

    # Load model
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # changing from batchnorm to sync batchnorm
    checkpoint_path = option.CHECKPOINT  # load the previous last model to retrain
    print(model)

    # Get the dataset glob it and load
    trn_paths = sorted(glob.glob(f'{option.INDIR}/training_data_*'))
    trn_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in trn_paths))
    # trn_dataset = LazyTorchDataset(f'{option.INDIR}/training_data_*')

    val_paths = sorted(glob.glob(f'{option.INDIR}/validation_data_*'))
    val_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in val_paths))
    # val_dataset = LazyTorchDataset(f'{option.INDIR}/validation_data_*')
    print("data read in")
    print(f"Train dataset: {len(trn_dataset)}")
    print(f"Validation dataset: {len(val_dataset)}")
    
    # here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=option.BS, num_workers=6, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=option.BS, num_workers=6, drop_last=True) 

    # Either recalculate the positive weight or take the old ones
    if option.CW:  # CW = calculate pos weights
        pos_weight = get_hetero_weight(trn_loader)
    else:
        pos_weight = {'t_nodes': torch.tensor(12.3976),'tt_edges': torch.tensor(481.3673), 'LCA': torch.tensor([2.5026e-01, 1.0037e+03, 3.4153e+02, 4.3406e+03]) }
        # {'t_nodes': tensor(12.3371), 'tt_edges': tensor(480.1262), 'LCA': tensor([2.5026e-01, 9.8915e+02, 3.4080e+02, 4.5415e+03])}
        # {'t_nodes': tensor(12.3976), 'tt_edges': tensor(481.3673), 'LCA': tensor([2.5026e-01, 1.0037e+03, 3.4153e+02, 4.3406e+03])}
    print(pos_weight)

    # Start the training here
    epochs = 30
    training(model, pos_weight, epochs, option.NGPU, trn_loader, val_loader, accumulate_grad_batches=option.GACC, checkpoint=checkpoint_path)
