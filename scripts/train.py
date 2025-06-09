from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.datasets.data_handler import DataHandler
from wmpgnn.model.model_loader import ModelLoader
from wmpgnn.trainers.gnn_trainer import GNNTrainer
from wmpgnn.trainers.hetero_gnn_trainer import HeteroGNNTrainer
import argparse

# load script arguments simply the config yaml
parser = argparse.ArgumentParser(description="Argument parser for the training.")
parser.add_argument("config", type=str, help="yaml config file for the training")
args = parser.parse_args()

# load config files with ConfigLoader class
print("Loading Config")
config_loader = ConfigLoader(f"config_files/{args.config}", environment_prefix="DL")

# load dataset with DataHandler
print(f"Loading Dataset {config_loader.get('dataset.data_type')}")
data_loader = DataHandler(config_loader)
data_loader.load_data()
train_loader = data_loader.get_train_dataloader()
val_loader = data_loader.get_val_dataloader()

# load model with ModelLoader
print(f"Initializing model {config_loader.get('model.type')}")
model_loader = ModelLoader(config_loader)
model = model_loader.get_model()

# initialized GNNTrainer or HeteroGNNTrainer
print("Training model")
add_bce = config_loader.get('loss.add_bce')
if config_loader.get('dataset.data_type') == "homogeneous":
    trainer = GNNTrainer(config_loader, model, train_loader, val_loader, add_bce = add_bce)
elif config_loader.get('dataset.data_type') == "heterogeneous":
    trainer = HeteroGNNTrainer(config_loader, model, train_loader, val_loader, add_bce=add_bce)
    beta_bce_pvs = config_loader.get('loss.beta_bce_pvs')
    trainer.set_beta_bce_pvs(config_loader.get('loss.beta_bce_pvs'))

trainer.set_beta_bce_nodes(config_loader.get('loss.beta_bce_nodes'))
trainer.set_beta_bce_edges(config_loader.get('loss.beta_bce_edges'))

epochs = config_loader.get('training.epochs')
learning_rate = config_loader.get('training.starting_learning_rate')
dropped_lr_epochs = config_loader.get('training.dropped_lr_epochs')

# Run training loop with nominal learning rate
print(f"Running {epochs} epochs with learning rate {learning_rate}")
trainer.train(epochs = epochs, learning_rate = learning_rate)


# Drop learning rate and continue for dropped_lr_epochs
if dropped_lr_epochs > 0:
    print(f"Running {dropped_lr_epochs} epochs with learning rate {learning_rate/10}")
    trainer.train(epochs=epochs+dropped_lr_epochs, starting_epoch=epochs,learning_rate=learning_rate/10)

# save model
model_file = config_loader.get("training.model_file")
print(f"Training finished. Saving model in {model_file}")
trainer.save_model(model_file)

# save dataframe with training and validation losses
trainer.save_dataframe("Final_full_graph_8block_32_epochs_weighted_message_passing_BCE.csv")

