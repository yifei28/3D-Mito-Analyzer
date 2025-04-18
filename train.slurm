#!/bin/bash
#SBATCH --job-name=train_mito          
#SBATCH --output=logs/train_mito_%j.out 
#SBATCH --error=logs/train_mito_%j.err   
#SBATCH --partition=gpuA40x4    
#SBATCH --gres=gpu:1                 
#SBATCH --mem=8G                     
#SBATCH --cpus-per-task=4            
#SBATCH --time=12:00:00               
#SBATCH --mail-type=END,FAIL         
#SBATCH --mail-user=bingkun3@illinois.edu 
#SBATCH --account=bczw-delta-gpu


module load gcc/11.4.0
export LD_LIBRARY_PATH=/sw/spack/deltas11-2023-03/apps/linux-rhel8-x86_64/gcc-8.5.0/gcc-11.4.0-yycklku/lib64:$LD_LIBRARY_PATH

source activate unet3d

cd /u/bfu1/3D-Mito-Analyzer

train3dunet --config train_config.yml