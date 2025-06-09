from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.performance.performance import Performance
import torch
import matplotlib.pyplot as plt
import argparse

# Turn interactive plotting off
plt.ioff()
# check for cuda or not use cuda by default if available
if torch.cuda.is_available():
    device = "cuda"
    cuda = True
    print("CUDA is available. Using GPU.")
else:
    device = "cpu"
    cuda = False
    print("CUDA not available. Using CPU.")


# load script arguments simply the config yaml
parser = argparse.ArgumentParser(description="Argument parser for the training.")
parser.add_argument("config", type=str, help="yaml config file for the training")
args = parser.parse_args()

# load config files with ConfigLoader class
print("Loading Config")
config_loader = ConfigLoader(f"config_files/{args.config}", environment_prefix="DL")

print("Using model: \n", config_loader.get('model'))

# set up performance class
# use config and set cuda=True or False
print(cuda)
perf = Performance(config_loader, cuda=cuda)

# evaluate performance for the inclusive scenario for 100 events with a final edge and node pruning cut of 0.2
# plot 10 perfect decay chains for reference
print("Running evaluate reco performance for 200 events with tight pruning ( > 0.2)")
perf.evaluate_reco_performance(event_max=200, pruning_cut=0.2, plot_perfect_decaychains=10)


# What about for an exclusive signal please use the commented lines below modifying the reference decay chains:
# signal_decay = {'daughters' : ['K+','K-','pi+','pi-','pi+','pi-'], 'mothers' : ['B0','D+','D-'] }
# cc_signal_decay = {'daughters' : ['K+','K-','pi+','pi-','pi+','pi-'], 'mothers' : ['B~0','D+','D-'] }
# ref_signal = (signal_decay, cc_signal_decay)
# perf.evaluate_reco_performance(event_max=5000, pruning_cut=0.2,plot_perfect_decaychains=10, ref_signal=ref_signal)
# note here we explicitly define too decays B0 and anti-B0 (B~0) with the final state daughters and any
# mothers in the decay chain B0, D+, D- where D+ -> K+ pi+ pi-

# The performance class also allows one to access

# For a ROC plot for node pruning you can run:
# Note edge pruning takes a lot longer due to the large number of edges.
print("Plotting Node pruning ROC curves with AUCs:")
perf.unset_pruning(layer=7) # we first unset the pruning we applied with perf.evaluate_reco_performance
true, pred = perf.evaluate_homog_track_pruning_performance(layers=[0,1,2,7],plot_roc=True, edge_pruning=False, batch_size=4)
# here layers is a list of layers where you want to plot the ROC performance for
# edge_pruning = True selects edge pruning where as False defaults to node pruning
# note this returns the true edge / node labels

# You can compute the LCA accuracy on the entire test sample with
# not this will consider any pruning selection set
acc = perf.evaluate_homog_lca_accuracy(batch_size=4)
print("LCA test accuracy: \n", acc)

# To set and unset pruning use the following:
# set edge and node pruning again, device is necessary for cpu inference due to some layer tensors by default on cuda
perf.set_edge_pruning(layer=7,cut=0.2, device=device)
perf.set_node_pruning(layer=7,cut=0.2, device=device)
perf.unset_pruning(layer=7) # unset last layer of pruning of a 8 layer HGNN


