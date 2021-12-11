#!/bin/bash
#SBATCH --job-name=UNET
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=8:00:00
#SBATCH --mail-type=END
#SBATCH --mail-user=zs1542@nyu.edu # put your email here if you want emails
#SBATCH --output=NominalV_%j.out
#SBATCH --error=NominalV_%j.err
#SBATCH --gres=gpu:1 # How much gpu need, n is the number

echo "Your NetID is: zs1542"
echo "Your environment is: py38"

module purge
module load anaconda3/2020.07

echo "start training"
source activate /scratch/zs1542/envs/py38/


cd /scratch/zs1542/CV-FinalProject/CVFramework
python /scratch/zs1542/CV-FinalProject/CVFramework/train_unet.py

echo "FINISH"