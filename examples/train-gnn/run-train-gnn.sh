#!/bin/bash
#
# Set the job name and wall time limit
#BSUB -J train-gnn
#BSUB -W 168:00
#
# Set the output and error output paths.
#BSUB -o  train-gnn-%J.o
#BSUB -e  train-gnn-%J.e
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

python train_gnn.py                                                         \
    --model-config-file         model-config.yaml                           \
    --model-config-file         data-config.yaml                            \
    --output-directory          "output"                                    \
    --n-epochs                  5000                                        \
    --n-gpus                    1                                           \
    --partial-charge-method     "am1"                                       \
    --data-cache-directory      "cached-data"                               