# Installation:

Currently for torch 2.4 however torch 2.5 is now availbale and should work fine.

cpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn
pip install dm-tree
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
```
note for gpu:
gpu: 
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

Will add a requirements file later.


# Neutrals framework:

## Installation :
You need Python 3.10 Cuda 12.4 and Pytorch 2.4.0 and the corresponding wheel for pytorch packages.

I suggest to create a conda virtual environnement, following this instructions :

```bash
conda create -n dfei_env python=3.10 -y
conda activate dfei_env
```

#### Install PyTorch with CUDA 12.4
```bash
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
```

#### Install PyG libraries
```bash
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html
```

#### Other dependencies
```bash
pip install scikit-learn dm-tree torch_geometric==2.6.1 uncertainties numpy pandas matplotlib mplhep pyyaml
```

The list of required packages are avalaible in the file `environment.yml` 

## Training
The HGNN can be trained and validated with this framework. You can use both simplified PYTHIA simulation or full MC simulation (for both magup or magdown polarities, or combined). You can choose to save the processed graphs and be able to load them in a later training in order to spare computing time. You can tune the parameters of the HGNN model and of the training. A option allows to balance the class (signal and background), by discarding random events in the majoritarian class. An early stopping condition with patience counter is used to stop th training when the validation loss function is not improving enough along a few epochs, according to the selected criterion.

### Config file
A config file must be filled with the parameters. 
You can take exemple on `./weighterd_MP_gnn/config_files/neutrals_hgnn_run3.yaml`.

Remarks on some parameters of the config file :
- `model.type` should be `neutral_heterognn`~and `dataset.type` should be `neutrals`
- Number of `gnn layers`, `mlp output_size`, `channels` and `layers` (idem for `weight mlp`) can be set
- `model.node_types` are `['chargedtree', 'neutrals']` and `model.edge_types` is `["chargedtree_neutrals"]` or `["chargedtree_neutrals", "neutrals_neutrals]`
- the `model.threshold` is the value that discriminate signal and backgrounds in predictions values
- the `model.dropout` set the fraction of dropout in the MLP modules of the GNN
- `dataset.evt_max_train` and `dataset.evt_max_train` select the number of events used
- The pre-processed graph can be saved if `dataset.save_graph` and loaded later if `dataset.load_graph`
- It is possible to train with balanced class (discarding random background neutral particles) if `dataset.balanced_classes`
- Training parameters can be modified (`training.epochs`, `training.batch_size`, `training.starting_learning_rate`, ...)
- You can save and load checkpoint during training with `training.load_checkpoint` and `training.save_checkpoint`
- An early stopping condition is used during training, set by  `training.early_stopping_patience` (patience coutner limit) and `training.early_stopping_min_delta` (minimum increasing of loss required)
- After the early stopping condition reached, there are still epoch with reduced learning rate (/10) set by `training.dropped_lr_epochs`.
- Each epoch used a random subsample of the train sample, the fraction of data excluded is set by `training.k_subsetRandomSampler` (exclude 1/k of total sample).
- If your sample si full MC, you can choose the polarity with `data.polarity`. If your sample is PYTHIA, simply set `data.polarity=PYTHIA`.

### Input files
To run you need to have the `input.npy` and `target.npy` files ready (for both training and validation datasets)
They are generated with the script `lhcbdfei/data_handling/input_formatting/cache_data_neutrals.py` on the main dfei GitLab repo
(https://gitlab.cern.ch/dfei/lhcbdfei/-/tree/ebornand_neutrals?ref_type=heads)
You need to specify the path to these files in the config file in `dataset.data_dir`
They must be stored in two folders named `training_dataset/` and `validation_dataset/`.

During the training of the HGNN, the data loading step process the datasets into graphs, if the corresponding option is activated, it is possible to store these processed graphs in pytorch format (.pt). To do so, set the parameter `save_graphs` to `True` in the `config_file` (see below). It is then possible to train the HGNN again directy using these graphs, instead of the numpay arrays input files, by setting the parameter `load_graphs` to `True` in the `config_file`. As the graph generation step is time consuming, this option is advantageous for the developpement. The graphs formats files will be stored in the same directory as your numpay array files, in a subfolder named `graphs` (or `graphs_nedges` if you activated edges between neutral particles).

### Train script
To train you model, call from the root folder (`./weighterd_MP_gnn/`) the following command (can be with another config file name): 

```bash
python -m scripts.train config_file.yaml
```

All outputs files and figures can be found in `./weighterd_MP_gnn/outputs/`

Different thresholds are used to compute the performance metrics:
-The default threshold 0.5
-A fixed signal efficiency 90% threshold 
-A fixed signal efficiency 99% threshold 
-An optimised threshold wrt the FOM S/sqrt(S+B)

Different informations are saved:
-The model (HGNN) in pytorch format (.pt)
-The parameter configuration
-All the performance metrics and confusion matrix for all epochs and all type of neutral particles
-Plots of the FOM optimisation and TPR against thresholds for the relevant epochs
-Plots of the ROC curve and of the predicition spectra for the relevant epochs.
-Plot of the loss function (train and val) along the training
-Plots of the different performance metrics (signal efficiency, background rejection, accuracy, balanced accuracy, precisions) for each thresholds along the epochs (and for each type of neutral particles separetely).


#### Remark  
You need GPUs to run the framework in this state. On lxplus, you can access some GPU nodes with :
```bash
ssh <your_user_name>@lxplus-gpu.cern.ch
```
Another option is to run it with condor on lxplus.
To do so, in the `submission_file.sub`, request GPUs with:
```
request_gpus = 1
```




