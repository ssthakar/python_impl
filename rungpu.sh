#!/bin/bash
#BSUB -n 1
#BSUB -W 48:00
#BSUB -q gpu

# #BSUB -R "select[a10]"
#BSUB -gpu "num=1:mode=shared:mps=yes"
#BSUB -o out.%J
#BSUB -e err.%J

# load cuda module
module load cuda/12.0
module load conda
source /home/ssthakar/.bashrc
conda activate /usr/local/usrapps/aicfd/ssthakar/conda_envs/ml_env 
# cd into the directory containing the run.sh file
CASE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $CASE_DIR

export CUDA_VISIBLE_DEVICES=0

python main.py
