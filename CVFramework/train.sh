#!/bin/bash
#SBATCH --job-name=unet_bseline
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=4:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=qz1086@nyu.edu # put your email here if you want emails
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number

module purge
module load anaconda3/2020.07

echo "start training"
source activate /scratch/qz1086/.penv

cd /scratch/zs1542/CV-FinalProject/CVFramework
python main.py

echo "FINISH"
echo "Have a Nice Day"