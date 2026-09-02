#!/usr/bin/env bash
# Introduction: run a single ID
# /home/huimin/Prepocessing/2026_03_10_adding_label.py

# 跑loop的command DT_M2105_386_before_comparing.pkl
for file in Preprocessing_output/preprocessing_output_2026_04_07/*_before_comparing.pkl; do
  name=$(basename "$file")
  echo "name : $name"
  id=${name%_before_comparing.pkl}
  echo "$id"

  python Prepocessing/2026_03_10_adding_label.py \
    --id $id \
    --input_folder_name  Preprocessing_output/preprocessing_output_2026_04_07/ \
    --output_folder_name Preprocessing_output/preprocessing_output_2026_04_07/ \
    --config_name preprocessing_config.yaml
done

#Usage:
#   salloc --time=96:00:00
#
#   conda activate ThesisEnv
#   cd /home/huimin
#   bash Prepocessing/adding_label_2026_04_07.sh > Prepocessing/Adding_Label_2026_04_07.txt