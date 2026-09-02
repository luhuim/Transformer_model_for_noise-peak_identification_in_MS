# /usr/bin/python3
"""
This part is to comparing the m/z and add anotation "0" or "1"

working directory(工作目录):/mnt/g/Lund_master_thesis/pyteomics

Input file:
    'DT_M2105_468_before_comparing.tsv'
Output files:
    'DT_M2105_468_after_comparing.tsv'

Requirement:
    pandas
    numpy
    math
Usage:
    nohup python -u 2023_10_23_adding_label.py > adding_label.log 2>&1 &
       
"""
#%% Importing the array module
import sys
import array
import pandas as pd
pd.set_option('display.max_columns', None)#show all columns in the dataframe
import pyteomics
from pyteomics import mass
import numpy as np
import re
np.set_printoptions(threshold=np.inf)#make the full array can be printed out

print("Package imported successfully.")
import importlib.metadata
print(f"pyteomics: {importlib.metadata.version('pyteomics')}")
print(f"pandas: {pd.__version__}")
print(f"python: {sys.version.split()[0]}") 
#%% 查看当前工作目录:/home/huimin 设置程序运行根目录
import os 
from pathlib import Path
print(os.getcwd()) 

# 设置程序运行根目录,自动生成的
# 获取当前script所在目录
current_dir = Path(__file__).resolve().parent
print("current_dir:",current_dir)       # current_dir: /home/huimin/Prepocessing

# 获取"py文件夹名字"(最后一层目录名)
program_dir = current_dir.name
print(f"program_dir:{program_dir}") #Prepocessing

# 获取 parent_dir 
parent_dir=current_dir.parent
print(f"parent_dir:{parent_dir}") # parent_dir:  /home/huimin

# 设置 program 的根目录为 parent_dir
os.chdir(parent_dir)
print("New working directory:", Path.cwd())

#%% Argv
import argparse
from pathlib import Path

"""
Argv:
    - id
    - pkl_dir = "2016_02_27_output"
    - out_dir = "2016_02_27_output"
    - config_name =""
Generate:
    pkl_name="{id}_before_comparing.pkl"  # in
"""

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--input_folder_name", required=True)
    p.add_argument("--output_folder_name", required=True)
    p.add_argument("--config_name", required=True)
    return p.parse_args()

def build_paths(args):
# import the variable from outside    
    #1 id
    id= f"{args.id}"
    print("id:",id)
    
    #2 input dir
    pkl_dir = Path(args.input_folder_name)
    print("pkl_dir:",pkl_dir)
    
    #3 output_dir
    out_dir = Path(args.output_folder_name)
    print("out_dir:",out_dir)

    #4 config file
    config_name = f"{args.config_name}"
    print("config_name:",config_name)

    # 构建(.pkl)路径,根据外部传入的变量 
    #input path
    pkl_path = pkl_dir / f"{id}_before_comparing.pkl"
    print("pkl_path:",pkl_path)
    #output path
    out_path = out_dir / f"{id}_label_added.pkl"
    print("out_path:",out_path)

    # 检查input (.pkl)文件是否存在(建议保留)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Input file not found: {pkl_path}")
    print(f"Input (.pkl) found: {pkl_path}")

    # 输出目录不存在就创建
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Return: id; input_file_path; output_file_path; config_file
    return id, pkl_path, out_path, config_name


def main():
    args = parse_args()
    id, pkl_path, out_path, config_name  = build_paths(args)

    print("ID:", id)
    print("input:", pkl_path)
    print("output:",out_path)
    print("config:",config_name)
    
    return id, pkl_path, out_path, config_name

if __name__ == "__main__":
    id, pkl_path, out_path, config_name= main()
    print("Finish")

#%% Read config file
import os
import yaml

# config_name = "preprocessing_config.yaml"
config_path = f"{program_dir}/{config_name}"

print(f"Config file path:{config_path}")

# Define ion_type 默认值
default_ppm = 20
# Define config 字典
config = {}

# 判断config文件是否存在
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {} #config is a dictionary
        # print(config)

# 获取 "ppm_scale"
ppm_scale = config.get( # 从config字典当中获取key,"ppm_scale"
                        "ppm_scale", 
                        default_ppm, #如果config当中没有这个key,就用默认值
                        )

print(f"ppm_scale:{ppm_scale}")

#%% Open the input file: "DT_M2105_386_before_comparing.pkl"
print(f"Open file:{pkl_path}") 
# Open the input file (.pkl)
before_comparing = pd.read_pickle(pkl_path)

# Check the shape of DataFrame
print("DataFrame shape:",before_comparing.shape)

# Check the output file path
print(f"Output path:{out_path}")

print(f"lenth:{len(before_comparing["ion_mass"][2])}")
print(before_comparing["ion_mass"][2])
print(f"lenth:{len(before_comparing["ion_mass"][100])}")
print(before_comparing["ion_mass"][100])

#%% Check the columns
before_comparing.columns
#%% Set the 20ppm
# before_comparing= pd.read_table(
#                     'DT_M2105_468_before_comparing.tsv',  # file name "before comparing"
#                     sep='\t',  # delimiter is tab
#                     usecols=[
#                             'ScanNr',       #int64
#                             'rtinseconds',  #float64
#                             'charge',       #float64, will be switched into 'int64'
#                             'm/z',          #object
#                             'Peptide',      #float64
#                             'pepmass',      #object
#                             'ion mass'      #object
#                             ],
#                                 )
# before_comparing['charge'] = before_comparing['charge'].astype('int64')

# Setting a scale
scale=20e-6 # this is 20ppm
# print("scale",type(scale))
#%% check the dataframe
print(before_comparing.head())
print(before_comparing.dtypes)

#%% Check
print(before_comparing["ion_mass"])

print(f"lenth:{len(before_comparing["ion_mass"][2])}")
print(before_comparing["ion_mass"][2])
print(f"lenth:{len(before_comparing["ion_mass"][3])}")
print(before_comparing["ion_mass"][3])



#%% Creating some new clolumns
#creating some empty comlumns
before_comparing["label"]=np.nan#create an empty column in "before comparing",storing lables
before_comparing["sum_of_peaks"]=np.nan#create an empty column in "before comparing",storing sum of peaks.
before_comparing["sum_of_1s"]=np.nan#create an empty column in "before comparing",storing sum of "1"s.
before_comparing["ration_of_1s"]=np.nan#create an empty column in "before comparing",storing ration of "1"s
before_comparing["sum_of_0s"]=np.nan

#set these columns as object
before_comparing["label"]=before_comparing["label"].astype('object')# set this comlom as object
before_comparing["sum_of_peaks"]=before_comparing["sum_of_peaks"].astype('object')
before_comparing["sum_of_1s"]=before_comparing["sum_of_1s"].astype('object')
before_comparing["ration_of_1s"]=before_comparing["ration_of_1s"].astype('object')
before_comparing["sum_of_0s"]=before_comparing["sum_of_0s"].astype('object')

#%% Loop of camparing
# Columns:'ScanNr', 'rtinseconds', 'pepmass', 'charge', 'm/z', 'intensity',
#         'DenovoPeptideSequence', 'SpectrumScore', 'clean_seq', 'ion_mass'
# Starting comparing the m/z with theoratical ion mass
for i in range (0,len(before_comparing)):#adding anotation row by row in dataframe "before_comparing"
    print(i)
    print(f"ScanNr:{before_comparing.loc[i,"ScanNr"]}")
    print(f"peptide sequence:{before_comparing.loc[i,'clean_seq']}")
    print(f"length of 'm/z': {len(before_comparing.loc[i,"m/z"])}")
    print(f"length of 'intensity':{len(before_comparing.loc[i,"intensity"])}")
    
    theory=before_comparing.loc[i,"ion_mass"] # 945
    print(f"length of 'ion_mass':{len(theory)}")
    
    actual=before_comparing.loc[i,'m/z']      # 78
    
    actual=np.vstack(actual)#make this array verticle
    print(f"vertical 'm/z':{actual}")

    # computing the ratio, and comparing this ratio with 20ppm
    # "c" is the absolute value of the ratio
    c=np.absolute((theory-actual)/theory) #calculate the ratio and get the absolute value
    #"c" is a matrix
    #print("c, shape:",c.shape) # c, shape: (78, 945)

    #print("Multiple and <20, shape:",(np.multiply(c,1000000) <= 20).shape)#(78, 945)
    #print("one array:",(np.multiply(c,1000000) <= 20)[1].shape) # (945,), one row
    #print("one array:",(np.multiply(c,1000000) <= 20)[1]) # the element is Bolean
    
    # compare and get the label, ppm_scale=20
    label=np.any(np.multiply(c,1000000) <= ppm_scale,axis=1) # multiply the ratio with 1 000 000, and then ask a question "whether there is any element (in one row in matrix 'c') <= 20?"
    # np.any() is to test whether any array element along a given axis evaluates to True.
    # a: Input array or object that can be converted to an array.
    # axis: Axis along which a logical OR reduction is performed.
    #       axis=1 is horizontal, one row
    # np.array.any()是"or"操作,任意一个元素为True,输出为True。

    #"lable" is an array of "true/False"
    #print("label shape:",label.shape) #(78,)
    label=label.astype(int)#change "true/false" into "1/0"#change "true/false" into "1/0"
    # print("label",label)

    # print("length of 'label':",len(label))
    # print("the sum of 1s:",np.sum(label))
    # print("ratio of 1s:",np.sum(label)/len(label))
    # print("sum of 0s:",len(label)-np.sum(label))
    
    # if np.count_nonzero(label >=1):
    #     print("*****************")
    #     print("lable:",label)
    #     print("the sum of 1:",np.sum(label))
    #     print(len(label))
    #     break
    # else:
    #     pass

    # lable=lable*1 # another method to convert "true/false" into "1/0"
    # print(lable)  # an array of "0 or 1"
    
    before_comparing.at[before_comparing.index[i], "label"] = label#adding this array into dataframe
    before_comparing.at[before_comparing.index[i], "sum_of_peaks"]=len(label)#"sum of peaks" is a column that show the total number of peaks
    # account
    before_comparing.at[before_comparing.index[i], "sum_of_1s"] = np.sum(label)#the sum of "1"s
    before_comparing.at[before_comparing.index[i], "ration_of_1s"] =np.sum(label)/len(label) #the ratio of "1"s among all peaks
    before_comparing.at[before_comparing.index[i], "sum_of_0s"] = len(label)-np.sum(label)
    print("One spectrum done.")
print("Finished all rows")
print("shape:",before_comparing.shape)
#%% Save in .pkl file.
before_comparing.to_pickle(out_path)
print(f"Saved in file: {out_path}")





