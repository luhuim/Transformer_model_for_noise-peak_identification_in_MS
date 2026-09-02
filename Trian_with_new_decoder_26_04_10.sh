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

#SBATCH --mail-user=hu6451lu-s@student.lu.se        	
#SBATCH --mail-type=ALL                             	
                                                    				# ALL means any state change
# Conda environment setting
module load Anaconda3
source config_conda.sh
conda activate new_env

# Enter into the project root directory
cd /path/to/project

# Run train.py 
python TrainingModel_2026_04_10.py



# Usage
# cd /path/to/project
# sbatch Trian_with_new_decoder_26_04_10.sh