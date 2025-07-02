import sys
import os
import glob
import re
from optparse import OptionParser
from tqdm import tqdm

import pandas as pd
import numpy as np

from itertools import chain

import torch
from torch_geometric.loader import DataLoader

from pytorch_lightning import Trainer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from wmpgnn.lightning.lightning_module import HGNNLightningModule
from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.util.functions import get_hetero_weight

from plot_helper import *

if __name__ == "__main__":
    # python eval.py --config ../../config_files/simple.yaml --indir /eos/user/y/yukaiz/DFEI_data/Bs_JpsiPhi --version 21 --cpt epoch-epoch=15.ckpt
    usage = "usage: %prog [options]"
    parser = OptionParser(usage)
    parser.add_option("", "--config", type=str, default=None,
                      dest="CONFIG", help="Config file path")
    parser.add_option("", "--indir", type=str, default=None,
                      dest="INDIR", help="Input directory where files are gobbled from")
    parser.add_option("", "--cpt", type=str, default=None,
                      dest="CHECKPOINT", help="Model checkpoint to be evaluated")
    parser.add_option("", "--version", type=int, default=None,
                      dest="VERSION", help="Lightning log version")
    (option, args) = parser.parse_args()
    if len(args) != 0:
        raise RuntimeError("Got undefined arguments", " ".join(args))

    """Load config file"""
    config_loader = ConfigLoader(option.CONFIG,
                                 environment_prefix="DL")  # One can include if it is included in the config loader and take that
    config = config_loader._load_config()
    config["mode"] = "Testing"

    """Load model"""
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # changing from batchnorm to sync batchnorm
    if config["eval"]["cpt"] == "None":
        print("Please specifiy checkpoint")
        exit()

    # Loading in the dataframe and the cpt model    
    version = config["eval"]["version"]
    # Getting the data frame
    dfs = []
    cpts = []
    bis_loss = []
    for v in version:  # combine the individual metrics for loss plotting, and select the best model from all iterations
        file_path = f"lightning_logs/version_{v}"
        df = pd.read_csv(f"{file_path}/metrics.csv")
        df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
        dfs.append(df)
        if config['eval']['cpt'] == "bis":
            cpt = glob.glob(f"lightning_logs/version_{v}/checkpoints/best*.ckpt")[0]
            cpts.append(cpt)
            match = re.search(r"val_combined_loss=(-?\d+\.\d+)", cpt)
            bis_loss.append(float(match[1]))
    if config['eval']['cpt'] == "bis":
        checkpoint_path = cpts[np.argmin(bis_loss)]
    else:
        try:
            checkpoint_path = f"lightning_logs/version_{config['eval']['cpt'][0]}/checkpoints/{config['eval']['cpt'][1]}"
        except:
            print("Wrong input format")

    df = pd.concat(dfs, ignore_index=True)

    # the pos weight arent used but are requried to be passed on
    pos_weight = {'t_nodes': torch.tensor(0.), 'tt_edges': torch.tensor(0.), 'LCA': torch.tensor([0., 0., 0., 0.]),
                  'frag': torch.tensor(0.), 'FT': torch.tensor([0., 0., 0.])}  # need to be load in from the hyperparams
    module = HGNNLightningModule.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        pos_weights=pos_weight,
        optimizer_class=torch.optim.Adam,
        optimizer_params={"lr": 1e-3},
        scheduler_params={},
        config=config
    )
    print(model)

    """Load data"""
    tst_paths = sorted(glob.glob(f"{config['data_dir']}/{config['eval']['sample']}/testing_data_*"))[:1]
    tst_dataset = []
    for path in tqdm(tst_paths, desc="Test dataset"):
        tst_dataset.extend(torch.load(path, weights_only=False))

    print("Data read in:")
    print(f"Test dataset       : {len(tst_dataset)}")
    tst_loader = DataLoader(tst_dataset[:500], batch_size=1, num_workers=6, drop_last=True)

    """Start evaluation"""
    # Plot the loss
    plot_loss(df, version[-1])
    # Obtain the LCA accuracy of the different classes as a function of the epochs
    plot_LCA_acc(df, version[-1])

    # Obtain the accuracy and peformance of the model
    trainer = Trainer(
        default_root_dir=f"lightning_logs/version_{version[-1]}",  # save the eval stuff in the dir of the model
    )
    trainer.test(module, dataloaders=tst_loader)

    # Load in signal df to calculate reco effiency and opposite side B finding
    signal_df = pd.read_csv(f"lightning_logs/version_{version[-1]}/signal_df_{config['eval']['sample']}.csv")
    sig_selbool = signal_df["SigMatch"] == 1
    signal_df = signal_df[sig_selbool]
    print(f"Number of signal B: {signal_df.shape[0]}")
    print(f"Number of perfect B: {np.sum(signal_df['PerfectReco'])}")
    print(f"Number of all particles B: {np.sum(signal_df['AllParticles'])}")
    print(f"Number of part reco B: {np.sum(signal_df['PartReco'])}")
    print(f"Number of none iso B: {np.sum(signal_df['NoneIso'])}")
