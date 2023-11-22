#!/bin/bash
 
#########################
## SLURM JOB COMMANDS ###
#########################
#SBATCH --partition=allcpu            ## or allgpu / cms / cms-uhh / maxgpu
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --job-name ROCeval           # give job unique name
#SBATCH --output ./run_files/ROCeval-%j.out      # terminal output
#SBATCH --error ./run_files/ROCeval-%j.err
#SBATCH --mail-type END
#SBATCH --mail-user sebastian.bieringer@desy.de
##SBATCH --constraint=P100
##SBATCH --exclude=max-cmsg007
 
##SBATCH --nodelist=max-cmsg[001-008]         # you can select specific nodes, if necessary

 
#####################
### BASH COMMANDS ###
#####################
 
## examples:
 
# source and load modules (GPU drivers, anaconda, .bashrc, etc)
source ~/.bashrc

# activate your conda environment the job should use
conda activate new_bayesconda
 
# go to your folder with your python scripts
cd /home/bierings/JetSurrogate/

echo $(date +"%Y%m%d_%H%M%S") $SLURM_JOB_ID $SLURM_NODELIST $SLURM_JOB_GPUS  >> cuda_vis_dev.txt

# run
python3 ROC_eval.py --path="$1"