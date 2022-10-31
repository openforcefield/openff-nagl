#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J search-hyperparameters
#BSUB -W 168:00
#
# Set the output and error output paths.
#BSUB -o  search-hyperparameters-%J.o
#BSUB -e  search-hyperparameters-%J.e
#
# Set any gpu options.
#BSUB -q gpuqueue
#BSUB -gpu num=1:j_exclusive=yes:mode=shared:mps=no:
#
#BSUB -M 32

# ===================== conda environment =====================
. ~/.bashrc
conda activate openff-nagl
conda env export > "${LSB_JOBNAME}-environment.yaml"

# ======================== script body ========================
mkdir output cached-data

python search_hyperparameters.py                                            \
    --model-config-file         gnn-config.yaml                             \
    --output-directory          "output"                                    \
    --n-epochs                  250                                         \
    --n-total-trials            300                                         \
    --n-gpus                    1                                           \
    --partial-charge-method     "am1"                                       \
    --data-cache-directory      "cached-data"                               \
    --output-config-file        "best-hyperparameters.yaml"
