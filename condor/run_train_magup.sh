#!/bin/bash
# Script to run the training job with GPU and Conda env

# Activate conda (ajuste le chemin si nécessaire)
# source /cvmfs/sft.cern.ch/lcg/views/LCG_104a/x86_64-centos7-gcc11-opt/setup.sh
# eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
source /afs/cern.ch/user/e/ebornand/miniconda3/etc/profile.d/conda.sh
conda activate dfeiGPUpy310

# Aller dans le répertoire de travail
cd /afs/cern.ch/user/e/ebornand/DFEI_HGNN/weighted_MP_gnn

nvidia-smi

# Lancer l'entraînement
python -m scripts.train neutrals_hgnn_run3_magup.yaml

