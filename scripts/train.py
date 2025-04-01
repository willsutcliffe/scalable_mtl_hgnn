import sys,os
sys.path.append(os.getcwd())

from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.datasets.data_handler import DataHandler
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.trainers.gnn_trainer import GNNTrainer
from wmpgnn.trainers.hetero_gnn_trainer import HeteroGNNTrainer
import torch
from torch import nn
import argparse

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

parser = argparse.ArgumentParser(description="Argument parser for the training.")
parser.add_argument("config", type=str, help="yaml config file for the training")
args = parser.parse_args()

print("Loading Config")
config_loader = ConfigLoader(f"config_files/{args.config}", environment_prefix="DL")

print(f"Loading Dataset {config_loader.get('dataset.data_type')}")
data_loader = DataHandler(config_loader)
data_loader.load_data()
train_loader = data_loader.get_train_dataloader()
val_loader = data_loader.get_val_dataloader()

print(f"Initializing model {config_loader.get('model.type')}")
model_loader = ModelLoader(config_loader)
model = model_loader.get_model()

print("Training model")
add_bce = config_loader.get('loss.add_bce')
if config_loader.get('dataset.data_type') == "homogeneous":
    trainer = GNNTrainer(config_loader, model, train_loader, val_loader, add_bce = add_bce)
elif config_loader.get('dataset.data_type') == "heterogeneous":
    trainer = HeteroGNNTrainer(config_loader, model, train_loader, val_loader, add_bce=add_bce)

epochs = config_loader.get('training.epochs')
learning_rate = config_loader.get('training.starting_learning_rate')
dropped_lr_epochs = config_loader.get('training.dropped_lr_epochs')

print(f"Running {epochs} epochs with learning rate {learning_rate}")
trainer.train(epochs = epochs, learning_rate = learning_rate)

if dropped_lr_epochs > 0:
    print(f"Running {dropped_lr_epochs} epochs with learning rate {learning_rate/10}")
    trainer.train(epochs=epochs+dropped_lr_epochs, starting_epoch=epochs,learning_rate=learning_rate/10)

model_file = config_loader.get("training.model_file")
# format the name with info from the config file
flatten_config = flatten_dict(config_loader.config)
model_file = model_file.format(**flatten_config)

print(f"Training finished. Saving model in {model_file}")
trainer.save_model(model_file)

csv_file = model_file.replace(".pt", ".csv")
trainer.save_dataframe(csv_file)

