# dFEI Weighted Message Passing GNN
Implementation of a weighted message passing GNN in pytorch 

## To Do

## TODO
- [] Modularize dataloading, training, eval with configs, classes and scripts
- [] Perform ablation trainings
- [] Perform 
- [] 
- [] Quantify inclusive reconstruction performance.

## OLDTO
- [x] Quantify inclusive reconstruction performance.
- [] Incorporate model in LCA evaluation script.
- [] Quantify exclusive reconstruction performance.
- [] Implement switch for message passing weights: no weights (w=1), sigmoid, softmax and perform systematic ablation study to understand influence of intermediate targets and weighted message passing.
- [] Investigate applicability of model to edge classification tasks like for CORA, OGB datasets etc
- [] Pruning performance vs speedup systematic study.
- [] Write paper



## Installation:

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


## Neutrals framework:

You need python 3.10 Cuda 12.4 and Pytorch 2.4.0 and the corresponding wheel for pytorch packages.

I suggest to create a conda virtual environnement, following this instructions :

conda create -n dfei_env python=3.10 -y
conda activate dfei_env

# Install PyTorch with CUDA 12.4
pip install torch==2.4.0+cu124 torchvision==0.19.0+cu124 torchaudio==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install PyG libraries
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cu124.html

# Other dependencies
pip install scikit-learn dm-tree torch_geometric==2.6.1 uncertainties numpy pandas matplotlib mplhep
