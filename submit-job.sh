#! /bin/bash

#SBATCH --job-name="HighwayPPO"
#SBATCH --partition=GPU
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=024:00:00
#SBATCH --gres=gpu:L40S:1
#SBATCH --cpus-per-task=32

module load cuda/12.4
module load cudnn/9.0.0-cuda12

cd "$SLURM_SUBMIT_DIR"

uv run main.py
