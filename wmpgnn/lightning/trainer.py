import sys
import os
import glob
from optparse import OptionParser
from tqdm import tqdm
import time

from multiprocessing import Pool
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

def load_file(path):
    files = torch.load(path, weights_only=False)
    print(f"file load: {path.split('/')[-1]}")
    return files

if __name__ == "__main__":
    # python trainer.py  --config  ../../config_files/lightning.yaml
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    # pivoting from option parser to full config file

    # Load config file
    config_loader = ConfigLoader(option.CONFIG, environment_prefix="DL")
    config = config_loader._load_config()
    config["mode"] = "Training"

    # Load model
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()
    checkpoint_path = config["training"]["cpt"]  # load the previous last model to retrain
    print(model)
    print("="*30)

    # Get the dataset glob it and load
    samples = config["training"]["sample"]
    print("Start reading in the data")
    print("Training:")
    start = time.time()
    trn_dataset = []
    for sample in samples:
        print(f"Loading {sample}")
        trn_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/training_data_*'))
        for path in tqdm(trn_paths, desc="Training dataset"):
            trn_dataset.extend(torch.load(path, weights_only=False))
    # trn_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in trn_paths))
    # trn_dataset = LazyTorchDataset(f'{option.INDIR}/training_data_*')

    print("Validation:")
    val_dataset = []
    for sample in samples:
        print(f"Loading {sample}")
        val_paths = sorted(glob.glob(f'{config["data_dir"]}/{sample}/validation_data_*'))
        for path in tqdm(val_paths, desc="Validation dataset"):
            val_dataset.extend(torch.load(path, weights_only=False))
    # val_dataset = list(chain.from_iterable(torch.load(p, weights_only=False) for p in val_paths))
    # val_dataset = LazyTorchDataset(f'{option.INDIR}/validation_data_*')
    end = time.time()

    print(f"data read in, time needed {(end - start):.2f}")
    print(f"Train dataset       : {len(trn_dataset)}")
    print(f"Validation dataset  : {len(val_dataset)}")
    print("="*30)

    # here we can check what kind of gpu it is to specify bs, also num_workers = num_cpu * 2
    trn_loader = DataLoader(trn_dataset, batch_size=config["training"]["batch_size"], num_workers=config["training"]["ncpu"]*2, drop_last=True) 
    val_loader = DataLoader(val_dataset, batch_size=config["training"]["batch_size"], num_workers=config["training"]["ncpu"]*2, drop_last=True) 

    # Either recalculate the positive weight or take the old ones
    print("Getting pos weight:")
    if config["training"]["cw"]:  # CW = calculate pos weights
        pos_weight = get_hetero_weight(trn_loader)
    else:
        pos_weight = {'t_nodes': torch.tensor(23.54585), 'tt_edges': torch.tensor(944.7520),
                      'LCA': torch.tensor([2.5026e-01, 9.7058e+02, 3.3759e+02, 4.2255e+03]),
                      'frag': torch.tensor(593.7332), 'FT': torch.tensor([16.1860,  0.3476, 16.2423])}
    print(pos_weight)
    print("="*30)

    # Start the training here
    epochs = 30
    training(model, pos_weight, epochs, config["training"]["ngpu"], trn_loader, val_loader, config, accumulate_grad_batches=config["training"]["gacc"], checkpoint_path=checkpoint_path)
