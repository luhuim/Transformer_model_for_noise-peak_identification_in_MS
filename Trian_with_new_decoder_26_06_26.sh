#!/bin/bash

#SBATCH --job-name=training_model_using_new_decoder	
#SBATCH --partition=gpua100				
#SBATCH --gres=gpu:1 					
#SBATCH --nodes=1					
#SBATCH --ntasks=1					
#SBATCH --cpus-per-task=48  				
#SBATCH --mem=491520MB      				        
#SBATCH --time=96:00:00            			
#SBATCH --output=slurm_output-%j.out			
#SBATCH --error=slurm_err_%j.txt
  	
#SBATCH --mail-type=ALL                             	
                                                    				# ALL means any state change
# Conda environment setting
module load Anaconda3
source config_conda.sh
conda activate new_env

# Enter into the absolute directory to run this python file 
cd /path/to/project

# Run train.py 
python TrainingModel_2026_06_26.py

# Usage
# cd /path/to/project
# sbatch Trian_with_new_decoder_26_06_26.sh