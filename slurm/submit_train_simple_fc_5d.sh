#!/bin/bash

#SBATCH --job-name=train_simple_fc_5d
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=20:00:00
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=tri.vt.nguyen@gmail.com

#SBATCH --output=logs/train_simple_fc_5d.out
#SBATCH --error=logs/train_simple_fc_5d.err


# Define path variables
# output directory
out_dir=$SCRATCH/train_simple_fc_5d/$SLURM_ARRAY_TASK_ID

# input directory: where the dataset is 
input_dir=$SCRATCH/dataset/m12i-lsr-0

# work directory: where running scripts are
work_dir=$WORK/gaia_merger_classification

# Define Python environment using Anaconda
# Note that you will also need to activate your Conda environment
# when submitting sbatch
conda activate fire-torch

# create output directory
mkdir -v -p $out_dir

# Define on exit job
function on_exit()
{
    # move output directory back to submission directory
    mv $out_dir $SLURM_SUBMIT_DIR
}
trap "on_exit" EXIT

# Run training scripts
cd $work_dir
srun python trainSimpleFC.py -i $input_dir -o $output_dir -t 5d -w\
 --store-val-output --n-max-files 1 -e 100 -b 1000 -N 8

# go back to submission directory
cd $SLURM_SUBMIT_DIR

# return success code
exit 0
