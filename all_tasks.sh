#!/bin/bash

folder="save/"
models=("WIKI_1.pt" "WIKI_2.pt" "WIKI_3.pt" "WIKI_4.pt" "WIKI_5512.pt")
tasks=("adv" "adv_adv" "adv_conjunction" "namepp" "nounpp" "nounpp_adv" "simple")

for model in ${models[@]}; do
    for task in ${tasks[@]}; do
        sbatch predictions.job $folder$model $task
    done
done