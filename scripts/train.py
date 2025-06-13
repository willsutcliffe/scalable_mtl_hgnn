import sys,os
sys.path.append(os.getcwd())

from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.datasets.data_handler import DataHandler
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.trainers.gnn_trainer import GNNTrainer
from wmpgnn.trainers.hetero_gnn_trainer import HeteroGNNTrainer
from wmpgnn.trainers.neutrals_hetero_gnn_trainer import NeutralsHeteroGNNTrainer
from wmpgnn.util.functions import select_epoch_indices

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import torch
from torch import nn
import argparse
import glob

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def GetWarmstartFile(checkpoint_path, name_query="checkpoint*"):
    """Search for the latest saved file in the given path."""
    # get list of checkpoint files in the output folder
    checkpoint_files = glob.glob(f'{checkpoint_path}{name_query}')
    checkpoint_file = None
    if len(checkpoint_files) > 0:
        checkpoint_file = checkpoint_files[-1]
        
    return checkpoint_file

parser = argparse.ArgumentParser(description="Argument parser for the training.")
parser.add_argument("config", type=str, help="yaml config file for the training")
args = parser.parse_args()

print("Loading Config")
config_loader = ConfigLoader(f"config_files/{args.config}", environment_prefix="DL")

print(f"Loading Dataset {config_loader.get('dataset.data_type')}")
data_loader = DataHandler(config_loader)
print('Data loader created')
data_loader.load_data()
train_loader = data_loader.get_train_dataloader()
val_loader = data_loader.get_val_dataloader()

print(f"Initializing model {config_loader.get('model.type')}")
model_loader = ModelLoader(config_loader)
model = model_loader.get_model()

model_file = config_loader.get("training.model_file")
# format the name with info from the config file
flatten_config = flatten_dict(config_loader.config)
model_file = model_file.format(**flatten_config)
# folder for the model and other outputs
if config_loader.get("dataset.balanced_classes", False) :
    model_file=model_file.replace(".pt", "_balanced.pt")
output_folder = f"outputs/{model_file.replace('.pt','')}/"
os.makedirs(output_folder, exist_ok=True)

print("Training model")
add_bce = config_loader.get('loss.add_bce')
if config_loader.get('dataset.data_type') == "homogeneous":
    trainer = GNNTrainer(config_loader, model, train_loader, val_loader, add_bce = add_bce)
elif config_loader.get('dataset.data_type') == "heterogeneous":
    trainer = HeteroGNNTrainer(config_loader, model, train_loader, val_loader, add_bce=add_bce)
elif config_loader.get('dataset.data_type') == "neutrals":
    threshold = config_loader.get('model.threshold')
    trainer = NeutralsHeteroGNNTrainer(config_loader, model, train_loader, val_loader, add_bce=add_bce, threshold=threshold)

checkpoint_path = f"{output_folder}"
print(f"Checkpoint path: {checkpoint_path}")
if config_loader.get('training.load_checkpoint'):
    checkpoint_file = GetWarmstartFile(checkpoint_path, name_query="checkpoint*")
    if checkpoint_file is not None:
        trainer.load_checkpoint(checkpoint_file)
        print(f"Checkpoint loaded from {checkpoint_file}")
    else:
        print("No checkpoint file found. Starting training from scratch.")


epochs = config_loader.get('training.epochs')
learning_rate = config_loader.get('training.starting_learning_rate')
dropped_lr_epochs = config_loader.get('training.dropped_lr_epochs')
min_delta = config_loader.get('training.early_stopping_min_delta')
patience=config_loader.get('training.early_stopping_patience')

print(f"Running {epochs} epochs with learning rate {learning_rate}")
save_checkpoint = config_loader.get('training.save_checkpoint')
trainer.train(epochs = epochs, learning_rate = learning_rate, early_stopping_patience=patience,
              min_delta=min_delta,starting_epoch=trainer.epoch_warmstart, save_checkpoint=save_checkpoint, checkpoint_path=checkpoint_path)

# In case of early stopping, we update the last epoch
if trainer.last_epoch > 0:
    last_epoch_early_stopping = trainer.last_epoch+1
else :
    last_epoch_early_stopping = epochs

if dropped_lr_epochs > 0:
    print(f"Running {dropped_lr_epochs} epochs with learning rate {learning_rate/10}")
    trainer.train(epochs=last_epoch_early_stopping+dropped_lr_epochs, starting_epoch=last_epoch_early_stopping, learning_rate=learning_rate/10)

print(f"Training finished. Saving model in {model_file}")
trainer.save_model(output_folder+model_file, save_config=True)

# csv_file = model_file.replace(".pt", ".csv")
# trainer.save_dataframe(output_folder+csv_file)

# make plots
plot_name = model_file.replace(".pt", "_loss.png")
trainer.plot_loss(output_folder+plot_name, show=False)

plot_name = model_file.replace(".pt", "_acc.png")
trainer.plot_accuracy(output_folder+plot_name, show=False)

plot_name = model_file.replace(".pt", "_eff.png")
trainer.plot_efficiency(output_folder+plot_name, show=False)

plot_name = model_file.replace(".pt", "_rej.png")
trainer.plot_rejection(output_folder+plot_name, show=False)

plot_name = model_file.replace(".pt", "_balacc.png")
trainer.plot_balanced_accuracy(output_folder+plot_name, show=False)

plot_name = model_file.replace(".pt", "_pre.png")
trainer.plot_precision(output_folder+plot_name, show=False)

for i in select_epoch_indices(last_epoch_early_stopping,dropped_lr_epochs,patience+2):
    plot_name = model_file.replace(".pt", f"_pred_epoch{i}.png")
    trainer.plot_predictions(output_folder, plot_name, epoch=i, show=False)
    plot_name = model_file.replace(".pt", f"_roc_auc_epoch{i}.png")
    trainer.plot_roc_auc(output_folder, plot_name, epoch=i, show=False)
    plot_name = model_file.replace(".pt", f"_threshold_epoch{i}.png")
    trainer.plot_tpr_thresholds(output_folder, plot_name, key_prefix='val', epoch=i, show=False)
    plot_name = model_file.replace(".pt", f"_fom_epoch{i}.png")
    trainer.plot_fom_vs_threshold(output_folder, plot_name, key_prefix='val', epoch=i, show=False)
#python scripts/train.py mp_gnn_run3.yaml | tee logs/homo.log

csv_file = model_file.replace(".pt", "_metrics.csv")
trainer.save_metrics(output_folder+csv_file)

