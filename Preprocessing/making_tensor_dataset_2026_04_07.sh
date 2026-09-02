#!/usr/bin/env bash
# This bash program is to run "2016_05_03_making_dataset.py"
#    /home/huimin/Prepocessing/2016_05_03_making_dataset.py

# 跑loop的command DT_M2105_386_label_added.pkl
for file in Preprocessing_output/preprocessing_output_2026_04_07/*_label_added.pkl; do
  name=$(basename "$file")
  echo "name : $name"
  id=${name%_label_added.pkl}
  echo "$id"

  python Prepocessing/2016_05_03_making_dataset.py \
    --id $id \
    --input_folder_name  Preprocessing_output/preprocessing_output_2026_04_07/ \
    --output_folder_name Preprocessing_output/preprocessing_output_2026_04_07/ \
    --config_name preprocessing_config.yaml
done

#Usage:
#   conda activate ThesisEnv
#   cd /home/huimin
#   bash Prepocessing/making_tensor_dataset_2026_04_07.sh > Prepocessing/making_tensor_dataset_2026_04_07.txt

