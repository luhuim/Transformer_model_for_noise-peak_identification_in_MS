#!/usr/bin/env bash

#Usage:
#   salloc --time=96:00:00
#   tmux new -s combine
#   conda activate ThesisEnv
#   cd /home/huimin
#   bash Prepocessing/Combine_mgf_csv.sh > Prepocessing/Combine_mgf_csv.txt

#
for file in Detroit562/*.csv; do
    name=$(basename "$file")
    echo "name : $name"
    id=${name%.csv}
    echo "$id"

    python Preprocessing/combine_mgf_and_percolator.py \
        --id "${id}" \
        --mgf_folder_name "Detroit562/mgf_folder" \
        --csv_folder_name "Detroit562" \
        --output_folder_name "Preprocessing_output"

done