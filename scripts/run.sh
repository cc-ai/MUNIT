#!/bin/bash
#SBATCH --job-name=testing_munit
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:titanxp:1
#SBATCH --output=synth_output.txt
#SBATCH --error=synth_error.txt
#SBATCH --ntasks=1

module load anaconda/3

source $CONDA_ACTIVATE

conda activate ccaienv

cd /network/home/raghupas/MUNIT/scripts/

python train.py --config ../configs/Final_test/FeatureDA+height30_seg.yaml
