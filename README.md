# Scalable Multi-Task Learning for Particle Collision Event Reconstruction with Heterogeneous GNNs

Github repo for “Scalable Multi-Task Learning for Particle Collision Event Reconstruction with Heterogeneous Graph Neural Networks”. 

Here we present code for a scalable Heterogeneous graph network with integrated pruning layers, 
which jointly determines if tracks originate from decay of beauty hadrons and associates each track
to a proton-proton collision point known as a primary vertex (PV). Both HGNN and GNN architectures
with node and edge pruning are available for benchmarking.

The paper is available at https://arxiv.org/abs/2504.21844


## Installation:

### Using `pip`

```bash
pip install -r requirements.txt
```

### Using `conda`

```bash
conda env create -n hgnn python=3.11 
conda activate hgnn
pip install -r requirements.txt
```

Note that for graphviz which is used for the performance you will need to install it:

Linux:
```bash
sudo apt-get install graphviz graphviz-dev
```

Mac:
```bash
sudo brew install graphviz
```

## Datasets

More information on the dataset is available in Dataset.md

The training, validation and test datasets are available on Zenodo:
https://zenodo.org/records/15584745


In particular for a default training of the model you should use the dataset
inclusive_training_validation_dataset
with ~40k training and ~10k validation events.

The corresonding test dataset (~10k) is: 
inclusive_test_dataset.tar.gz

Meanwhile, for several exclusive decays we also provide test samples with more detail in Dataset.md
## Config files

The training and inference of the HGNN and GNN models are controlled by yaml config files 
located in config_files. These configure model and training hyperparameters and the model
to load in the case of inference.

For training set the relevant path to the datasets described above. 

## Training

In order to train the HGNN model you can run the following command:
```bash 
python3 -m scripts.train  small_hetero_gnn.yaml 
```
meanwhile the GNN training can be run with
```bash 
python3 -m scripts.train  small_mp_gnn.yaml 
```


## Inference and performance

For model inference and assessing performance we provide scripts for the HGNN:
```bash 
python3 -m scripts.hetero_inference  test_heteromp_gnn_inference.yaml 
```
and GNN:
```bash 
python3 -m scripts.homog_inference  test_mp_gnn_inference.yaml 
```


