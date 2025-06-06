import sys
import os
import glob
from optparse import OptionParser
from tqdm import tqdm

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
    config_loader = ConfigLoader(option.CONFIG, environment_prefix="DL")  # One can include if it is included in the config loader and take that
    config_file = config_loader._load_config()  # pass this to the lightning module

    """Load model"""
    model_loader = ModelLoader(config_loader)
    model = model_loader.get_model()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)  # changing from batchnorm to sync batchnorm
    if option.CHECKPOINT is None:
        print("Please specifiy checkpoint")
        exit()
    checkpoint_path = F"lightning_logs/version_{option.VERSION}/checkpoints/{option.CHECKPOINT}"   # load the previous last model to retrain
    # the pos weight arent used but are requried to be passed on
    pos_weight = {'t_nodes': torch.tensor(12.2783), 'tt_edges': torch.tensor(612.2586), 'LCA': torch.tensor([2.5020e-01, 4.8452e+02, 8.9243e+02, 1.2172e+04]), 'frag': torch.tensor(0.6540), 'FT': torch.tensor([0.4930, 0.0106, 0.4964])} # need to be load in from the hyperparams
    module = HGNNLightningModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            model=model,
            pos_weights=pos_weight,
            optimizer_class=torch.optim.Adam,
            optimizer_params={"lr": 1e-3},
            version=option.VERSION,
            ref_signal=option.INDIR.split('/')[-1]
        )
    print(model)

    """Load data"""
    tst_paths = sorted(glob.glob(f'{option.INDIR}/testing_data_*'))[:1]
    tst_dataset = []
    for path in tqdm(tst_paths, desc="Validation dataset"):
        tst_dataset.extend(torch.load(path, weights_only=False))

    print("Data read in:")
    print(f"Test dataset       : {len(tst_dataset)}")
    tst_loader = DataLoader(tst_dataset[:500], batch_size=1, num_workers=6, drop_last=True)

    """Start evaluation"""
    # Getting the data frame
    version = option.VERSION
    file_path = f"lightning_logs/version_{version}"
    df = pd.read_csv(f"{file_path}/metrics.csv")
    df = df.groupby('epoch').agg(lambda x: x.dropna().iloc[0] if not x.dropna().empty else None).reset_index()
    
    # Plot the loss
    plot_loss(df, version)
    # Obtain the LCA accuracy of the different classes as a function of the epochs
    plot_LCA_acc(df, version)

    # Obtain the accuracy and peformance of the model
    trainer = Trainer(
        default_root_dir=f"lightning_logs/version_{version}", # save the eval stuff in the dir of the model
    )
    trainer.test(module, dataloaders=tst_loader)

    

