#!/usr/bin/env bash
# Introduction: run a single ID
# Usage:
#     cd /home/huimin/Prepocessing/
#     bash Run_test_file.sh > combine_mgf_and_percolator_output.txt

# ##################################### combine_mgf_and_percolator.py
# 跑单个文件的command
# python combine_mgf_and_percolator.py \
#   --id DT_M2106_048 \
#   --mgf_folder_name Detroit562/mgf_folder/ \
#   --csv_folder_name Detroit562/ \
#   --output_folder_name Detroit562/2016_02_27_output/

###########################################test.py
# 跑单个文件的command
# python test.py \
#   --id DT_M2106_048 \
#   --input_folder_name  Detroit562/2016_02_27_output/ \
#   --output_folder_name Detroit562/2016_02_27_output/ \
#   --config_name preprocessing_config.yaml

# 跑"2026_04_07_calculate_ion_mass.py"这个program的command
for file in Preprocessing_output/*_before_annotation.pkl; do
  name=$(basename "$file")
  echo "name : $name"
  id=${name%_before_annotation.pkl}
  echo "$id"
  
  python Prepocessing/2026_04_07_calculate_ion_mass.py \
    --id $id \
    --input_folder_name  Preprocessing_output/ \
    --output_folder_name Preprocessing_output/preprocessing_output_2026_04_07/ \
    --config_name preprocessing_config.yaml
done
#Usage:
#   salloc --time=96:00:00
#
#   conda activate ThesisEnv
#   cd /home/huimin
#   bash Prepocessing/Theoretical_fragment_ion_2026_04_07.sh > Prepocessing/Theoretical_fragment_ion_2026_04_07.txt







