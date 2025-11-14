import sys,os
hgnnroot = "../../weighted_MP_gnn/"
sys.path.append(hgnnroot) #path to the scalable_mtl_hgnn repository root

from wmpgnn.configs.config_loader import ConfigLoader
from wmpgnn.performance.performance import Performance
import torch
import matplotlib.pyplot as plt
import argparse
from numpy import linspace

from yaml import safe_load
with open("archive_of_paths.yaml", 'r') as f:
    ArchivePaths = safe_load(f)
    home = ArchivePaths.get("home", ".")
    if home != ".":
        for key in ArchivePaths:
            if key != "home":
                ArchivePaths[key] = home+ArchivePaths[key]

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
config_loader = ConfigLoader(f"{hgnnroot}/wmpgnn/config_files/{args.config}", environment_prefix="DL")

print("Using model: \n", config_loader.get('model'))
gnnLayers = config_loader.get('model.gnn_layers')
last_layer_index = int(gnnLayers) - 1 # last layer of the HGNN
layers_indexes =  [round(l) for l in linspace(0, last_layer_index,5)] # list of layer indexes for plotting
print("Inspecting layers: ", layers_indexes)
# set up performance class
# use config and set cuda=True or False
print(cuda)
perf = Performance(config_loader, cuda=cuda)

if False:
    print("Plotting node features for correct and misclassified tracks")
    perf.plot_misclassified_hetero_features(layers=[0,last_layer_index], batch_size=4, edge_pruning=False, pv_tr_edges = False, show_plots=False)
    print("Plotting edge features for correct and misclassified edges")
    perf.plot_misclassified_hetero_features(layers=[0,last_layer_index], batch_size=4, edge_pruning=True, pv_tr_edges = False, show_plots=False)

if True:
    # evaluate performance for the inclusive scenario for 100 events with a final edge and node pruning cut of 0.2
    # plot 10 perfect decay chains for reference
    pruning_cut = 0.5
    print(f"Running evaluate reco performance for 200 events with tight pruning ( > {pruning_cut})")
    perf.evaluate_reco_performance(event_max=200, pruning_cut=pruning_cut, layer_indx=last_layer_index, plot_perfect_decaychains=10)


# What about for an exclusive signal please use the commented lines below modifying the reference decay chains:
# signal_decay = {'daughters' : ['K+','K-','pi+','pi-','pi+','pi-'], 'mothers' : ['B0','D+','D-'] }
# cc_signal_decay = {'daughters' : ['K+','K-','pi+','pi-','pi+','pi-'], 'mothers' : ['B~0','D+','D-'] }
# ref_signal = (signal_decay, cc_signal_decay)
# perf.evaluate_reco_performance(event_max=5000, pruning_cut=0.2,plot_perfect_decaychains=10, ref_signal=ref_signal)
# note here we explicitly define too decays B0 and anti-B0 (B~0) with the final state daughters and any
# mothers in the decay chain B0, D+, D- where D+ -> K+ pi+ pi-

# The performance class also allows one to access
if False:
    # For a ROC plot for node pruning you can run:
    # Note edge pruning takes a lot longer due to the large number of edges.
    print("Plotting Node pruning ROC curves with AUCs:")
    perf.unset_pruning(layer=last_layer_index) # we first unset the pruning we applied with perf.evaluate_reco_performance
    true, pred = perf.evaluate_hetero_track_pruning_performance(layers=layers_indexes,plot_roc=True, edge_pruning=False, batch_size=4)
    # here layers is a list of layers where you want to plot the ROC performance for
    # edge_pruning = True selects edge pruning where as False defaults to node pruning
    true, pred = perf.evaluate_hetero_track_pruning_performance(layers=layers_indexes,plot_roc=True, edge_pruning=True, batch_size=4)

    # note this returns the true edge / node labels
if False:
    # You can compute the LCA accuracy on the entire test sample with
    # not this will consider any pruning selection set
    n_LCA = config_loader.get('model.LCA_classes', default=4)  # number of LCA classes
    acc = perf.evaluate_hetero_lca_accuracy(batch_size=4, nLCA=n_LCA)
    print("LCA test accuracy: \n", acc)

if False:
    # To set and unset pruning use the following:
    # set edge and node pruning again, device is necessary for cpu inference due to some layer tensors by default on cuda
    perf.set_edge_pruning(layer=last_layer_index,cut=0.2, device=device)
    perf.set_node_pruning(layer=last_layer_index,cut=0.2, device=device)
    perf.unset_pruning(layer=last_layer_index) # unset last layer of pruning of a 8 layer HGNN

    # track PV association with the HGNN is assessed with
    # note by default b_tracks = True
    acc, npvs, assoc = perf.evaluate_pv_association(batch_size=4, b_tracks=True)
    print("PV missassociation of B tracks: \n", acc)

    # Here acc is the an average accuracy over all tracks, assoc is a list of booleans
    # which denote if the PV was associated or not correctly.
    # Finally npvs is the multiplicity of the event for given PV.
    # One can also run the PV association for a custom use case by using the loop logic in the function.
