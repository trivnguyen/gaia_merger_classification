#!/bin/bash

#SBATCH --job-name=train_simple_fc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=20:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tri.vt.nguyen@gmail.com

#SBATCH --output=logs/train_simple_fc.out
#SBATCH --error=logs/train_simple_fc.err

OUT_DIR=$SCRATCH/train_simple_fc/$SLURM_ARRAY_TASK_ID  # output directory
INPUT_DIR=$SCRATCH/dataset/m12i-lsr-0  # input directory
WORK_DIR=$WORK/gaia_merger_classification  # working directory

# activate conda environment
conda activate fire-torch

# create output directory
mkdir -v -p $OUT_DIR

# define on exit job
function on_exit()
{
    mv $OUT_DIR $SLURM_SUBMIT_DIR
}
trap "on_exit" EXIT

# run training script
cd $WORK_DIR
srun python trainSimpleFC.py -i $INPUT_DIR -o $OUT_DIR -e 100 --n-max-files 1
cd $SLURM_SUBMIT_DIR

# move output back to slurm directory
#mv $OUT_DIR $SLURM_SUBMIT_DIR
