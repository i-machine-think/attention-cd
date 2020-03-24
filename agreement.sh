#!/bin/bash

#SBATCH --job-name=SHA-AGRE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0:10:00
#SBATCH --mem=16000M
#SBATCH --partition=gpu_short
#SBATCH --gres=gpu:1

python get_agreement_accuracy_for_contrast.py -ablation-results output/nounpp.abl -info stimuli/nounpp.info -condition number_1=singular number_2=singular
python get_agreement_accuracy_for_contrast.py -ablation-results output/nounpp.abl -info stimuli/nounpp.info -condition number_1=singular number_2=plural
python get_agreement_accuracy_for_contrast.py -ablation-results output/nounpp.abl -info stimuli/nounpp.info -condition number_1=plural number_2=singular
python get_agreement_accuracy_for_contrast.py -ablation-results output/nounpp.abl -info stimuli/nounpp.info -condition number_1=plural number_2=plural