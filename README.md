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
pip install scikit-learn dm-tree torch_geometric==2.6.1 uncertainties numpy pandas matplotlib mplhep
```

The list of required packages are avalaible in the file `environment.yml` 

## Training

### Config file
A config file must be filled with the parameters. 
You can take exemple on `./weighterd_MP_gnn/config_files/neutrals_hgnn_run3.yaml`.

Comments on some parameters of the config file :
- `model.type` should be `neutral_heterognn`~and `dataset.type` should be `neutrals`
- Number of `gnn layers`, `mlp output_size`, `channels` and `layers` (idem for `weight mlp`) can be set
- `model.node_types` are `['chargedtree', 'neutrals']` and `model.edge_types` is only `["chargedtree_neutrals"]`
- the `model.threshold` is the value that discriminate signal and backgrounds in predictions values
- `dataset.evt_max_train` and `dataset.evt_max_train` select the number of events used
- The pre-processed graph can be saved if `dataset.save_graph` and loaded later if `dataset.load_graph`
- It is possible to train with balanced class (discarding random background neutral particles) if `dataset.balanced_classes`
- Trainig parameters can be modified (`training.epochs`, `training.batch_size`, `training.starting_learning_rate`, ...)
- You can save and load checkpoint during training with `training.load_checkpoint` and `training.save_checkpoint`


### Input files
To run you need to have the `input.npy` and `target.npy` files ready (for both training and validation datasets)
They are generated with the script `lhcbdfei/data_handling/input_formatting/cache_data_neutrals.py` on the main dfei GitLab repo
(https://gitlab.cern.ch/dfei/lhcbdfei/-/tree/ebornand_neutrals?ref_type=heads)
You need to specify the path to these files in the config file in `dataset.data_dir`
They must be stored in two folders named `training_dataset/` and `validation_dataset/`.

### Train script
To train you model, call from the root folder (`./weighterd_MP_gnn/`) the following command (can be with another config file name): 

```bash
python -m scripts.train neutrals_hgnn_run3.yaml
```

All outputs files and figures can be found in `./weighterd_MP_gnn/outputs/`



