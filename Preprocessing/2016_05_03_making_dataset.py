# /usr/bin/python3
"""
Introduction:


working directory(工作目录):

Input file:

Output files:


Requirement:
    pandas
    numpy
    math
Usage:

       
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
    pkl_path = pkl_dir / f"{id}_label_added.pkl"
    print("pkl_path:",pkl_path)
    #output path
    out_path = out_dir / f"{id}_tensor_dataset.pkl"
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

#%% Open the pkl file DT_M2106_039_label_added.pkl
# pkl_dir = "Preprocessing_output/preprocessing_output_2026_05_03/"
# pkl_name= "DT_M2106_039_label_added.pkl" 
# 构建路径
# pkl_path= pkl_dir + pkl_name

print(f"Open file:{pkl_path}") 
# Open the input file (.pkl)
labeled_df = pd.read_pickle(pkl_path)
print(f"DataFrame shape:{labeled_df.shape}")

#%% Check the shape of DataFrame
print("DataFrame shape:",labeled_df.shape)

print("columns:",labeled_df.columns)
print(labeled_df.head())
print(labeled_df.iloc[20])
for r in labeled_df.iloc[20]:
    print(type(r))

#%% 把"intensity"归一化:在函数“making_array()当中添加intensity归一化的内容)
from sklearn.preprocessing import MinMaxScaler
def normalize_intensity(df,intensity_col):
    """
    对 DataFrame 中的 intensity 列中的数组进行归一化处理。
    
    Parameters:
        - df (pandas.DataFrame): 包含质谱数据的 DataFrame。
        - intensity_colu (str): DataFrame 中表示 intensity 的列名。

    Returns:
        pandas.DataFrame: 归一化后的 DataFrame。
    """
    # 初始化归一化器
    scaler = MinMaxScaler()
    
    df[intensity_col] = df[intensity_col].apply(    
            lambda x: scaler.fit_transform(x.reshape(-1, 1)).flatten() if len(x) > 0 else x
            )
    
    return df

labeled_df_2=normalize_intensity(labeled_df,"intensity")
print(labeled_df_2)

# Check the intensity
labeled_df_2.loc[20]["intensity"].sum()

#%% 计算pepmess的(H+)的mass
import pandas as pd

# 获取氢离子的质量
proton_mass=float(1.0073)

def add_protonated_mass(df,             # Dataframe that include spectrum infomation
                        pepmass_column, # column name of "PEPMASS" 
                        charge_column,  # column name of "charge"
                        new_column='protonated_mass',# the new column name
                        hydrogen_mass=float(1.0073)# default mass of "H+"
                        ):
    """
    计算每个质谱带一个正电荷时的质量，并将结果放入一个新的列中。
    
    Parameters:
        - df (pandas.DataFrame): 包含质谱数据的 DataFrame。
        - pepmass_column (str):  DataFrame 中表示中性质量的列名。
        - charge_column (str):   DataFrame 中表示电荷数的列名。
        - new_column (str):      新列的名称，用于存储带一个正电荷时的质量。
        - hydrogen_mass (float): 氢离子的默认质量
        
    Returns:
        pandas.DataFrame: 带有新列的 DataFrame。
    """
    
    #Here is the known information
    M_ion=df[pepmass_column] # pepmass
    n=df[charge_column]      # charge
    H_ion=hydrogen_mass      # H 
    
    #calculate neuro mass of percurs
    neuron_mass=M_ion*n-n*H_ion # pepmass*charge-n*H
    #calculate protonated mass of percurs
    M_protonated= neuron_mass+ H_ion
    
    #Add protonated mass in new column
    df[new_column]=M_protonated

    return df

# 调用函数，将计算的结果放入新的列 'protonated_mass'
labeled_df_3 = add_protonated_mass(
                                labeled_df_2,#Dataframe that include spectrum infomation
                                "pepmass", #column name of "pepmass" 
                                "charge",  #column name of "charge"
                         )

print(labeled_df_3[["pepmass","protonated_mass"]])
print(f"%%%%Columns of labeled_df_3:{labeled_df_3.columns}")
#%% 过滤掉"m/z"列中大于等于'protonated_mass'值的peak,并将符合条件的m/z和intensity存入新的列中
def filter_peaks(df,                    # 包含质谱数据的DataFrame,包括M+离子的mass
                 mz_column,             # DataFrame 中表示 m/z 的列名
                 intensity_column,      # DataFrame 中表示 intensity 的列名
                 label_column,          # DataFrame 中表示 label 的列名
                 protonated_mass_column # DataFrame 中表示 protonated_mass 的列名
                 ):
    """
    过滤掉 m/z 列中大于等于 protonated_mass 列值的峰，并将符合条件的 m/z 和 intensity 存入新的列中。
    
    Parameters:
    df (pandas.DataFrame): 包含质谱数据的 DataFrame。
    mz_column (str): DataFrame 中表示 m/z 的列名。
    intensity_column (str): DataFrame 中表示 intensity 的列名。
    protonated_mass_column (str): DataFrame 中表示 protonated_mass 的列名。
    
    Returns:
    pandas.DataFrame: 经过过滤的 DataFrame,包含新列存储符合条件的 m/z 和 intensity。
    """
    # 计算新的 mz; intensity和Label列
    def filter_row(mz_array, intensity_array, label_array, protonated_mass):
        mask = mz_array < protonated_mass # 符合条件的掩码mask
        return mz_array[mask], intensity_array[mask], label_array[mask]

    # 添加新的列
    df['filtered_mz'] = df.apply(
                                    #每一行(row)
                                    # lambda row: np.append(filter_row(row[mz_column], # mz_array
                                    lambda row: filter_row(row[mz_column], # mz_array
                                                           row[intensity_column], # intensity_array
                                                           row[label_column],# label_array
                                                           row[protonated_mass_column])[0],
                                                          # None),
                                    axis=1)
    
    df['filtered_intensity'] = df.apply(
                                    #每一行(row)
                                    # lambda row: np.append(filter_row(row[mz_column], # mz_array 新添加的
                                    lambda row: filter_row(row[mz_column], # mz_array#以前的那一行
                                                           row[intensity_column], # intensity_array
                                                           row[label_column],# label_array
                                                           row[protonated_mass_column])[1],
                                                          # None),# new row 
                                    axis=1)
    
    df['filtered_label'] = df.apply(
                                    lambda row: np.insert(
                                            filter_row(row[mz_column], 
                                                       row[intensity_column], 
                                                       row[label_column], 
                                                       row[protonated_mass_column])[2], 
                                            0, 1), #在index=0处添加整数'1'
                                            axis=1)
    
    return df
#%% 应用function过滤peaks, 过滤数据
labeled_df_4 = filter_peaks(
                            labeled_df_3, 
                           'm/z', 
                           'intensity', 
                           'label',
                           'protonated_mass'
                           )

print(labeled_df_4)

#%% Creating some new clolumns in "labeled_df_4"

# Create an empty column in "before comparing",storing lables
labeled_df_4["filtered_sum_of_labels"]=np.nan#create an empty column in "before comparing",storing sum of peaks.
labeled_df_4["filtered_sum_of_1s"]=np.nan#create an empty column in "before comparing",storing sum of "1"s.
labeled_df_4["filtered_ration_of_1s"]=np.nan#create an empty column in "before comparing",storing ration of "1"s
labeled_df_4["filtered_sum_of_0s"]=np.nan
labeled_df_4["filtered_changed"]=np.nan
#set these columns as object
labeled_df_4["filtered_sum_of_labels"] = labeled_df_4["filtered_sum_of_labels"].astype('object')# Done
labeled_df_4["filtered_sum_of_1s"] = labeled_df_4["filtered_sum_of_1s"].astype('object')#Done
labeled_df_4["filtered_ration_of_1s"] = labeled_df_4["filtered_ration_of_1s"].astype('object')
labeled_df_4["filtered_sum_of_0s"] = labeled_df_4["filtered_sum_of_0s"].astype('object')#Done
labeled_df_4["filtered_changed"] = labeled_df_4["filtered_changed"].astype("object")#Done

#%% Fill some sum values 
# filtered_label 这个label开头有固定的"1"
labeled_df_4["filtered_sum_of_labels"] = labeled_df_4["filtered_label"].apply(len)# 这个label开头有固定的"1"
labeled_df_4["filtered_changed"]= (
                        labeled_df_4["filtered_sum_of_labels"] - 1 - labeled_df_4["sum_of_peaks"]
                                    )
labeled_df_4["filtered_sum_of_1s"] = labeled_df_4["filtered_label"].apply(lambda x: sum(x) - 1) # 这个label开头有固定的"1"
labeled_df_4["filtered_sum_of_0s"] = labeled_df_4["sum_of_peaks"] - labeled_df_4["filtered_sum_of_1s"]
labeled_df_4["filtered_ration_of_1s"]= labeled_df_4["filtered_sum_of_1s"]/labeled_df_4["sum_of_peaks"]

# print()
print(f"filtered_label:        \n{labeled_df_4["filtered_sum_of_labels"]}")
print(f"original label:        \n{labeled_df_4["sum_of_peaks"]}")
# print(f"sum_of_peaks_using column'filtered_mz':\n{labeled_df_4["filtered_mz"].apply(len)}")
print(f"'filtered_changed':    \n{labeled_df_4["filtered_changed"]}")
print(f"'filtered_sum_of_1s':  \n{labeled_df_4["filtered_sum_of_1s"]}")
print(f"'filtered_sum_of_0s':  \n{labeled_df_4["filtered_sum_of_0s"]}")
print(f"'filtered_ration_of_1s'\n{labeled_df_4["filtered_ration_of_1s"]}")

#%%再写一个function: 把array转化成二维tensor，针对 "m/z"，"intensity"，"label"
import torch
from torch import nn
def making_tensor(df):
    """
    Process the DataFrame to convert specific columns from strings to arrays and then to tensors.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing columns 'm/z', 'intensity',  'label',"pepmass"and "charge".
    
    Returns:
        pd.DataFrame: The processed DataFrame with converted columns.
    """
    required_columns = ['filtered_mz', 'filtered_intensity', 'filtered_label',"pepmass","charge"]
    
    # 检查 DataFrame 是否包含所需的列
    for column in required_columns:
        if column not in df.columns:
            raise ValueError(f"DataFrame is missing required column: {column}")
    
    # 将 'm/z' 列从array，再转化为一维tensor，最后转化为二维tensor
    df['filtered_mz'] = df['filtered_mz'].apply(lambda x: torch.tensor(x))
    df['filtered_mz'] = df['filtered_mz'].apply(lambda x: torch.unsqueeze(x, 0).float())

    # 将 'intensity' 列从array，再转化为一维tensor，最后转化为二维tensor
    df['filtered_intensity'] = df['filtered_intensity'].apply(lambda x: torch.tensor(x))
    df['filtered_intensity'] = df['filtered_intensity'].apply(lambda x: torch.unsqueeze(x, 0).float())

    # 将 'label' 直接从array转化为二维tensor
    df['filtered_label'] = df['filtered_label'].apply(lambda x: torch.tensor(x).unsqueeze(0))#直接从array转化为二维tensor

    #将"pepmass"那一列变成float格式
    df["pepmass"]=df["pepmass"].astype(float) #<class 'numpy.float64'>
    #将"charge"那一列变成int格式
    df["charge"]=df["charge"].astype(int)     #<class 'numpy.int64'>


    return df
#%% Use the function, transform array into tensor
labeled_df_5 = making_tensor(labeled_df_4)
#%% Check the result
print(labeled_df_5.dtypes)
print(labeled_df_5["filtered_mz"][67].shape,
      labeled_df_5["filtered_intensity"][67].shape,
      labeled_df_5["filtered_label"][67].shape,
      )             
print(
        labeled_df_5["filtered_mz"][67],"\n",
        labeled_df_5["filtered_intensity"][67],"\n",
        labeled_df_5["filtered_label"][67],"\n",
        labeled_df_5["pepmass"][67],"\n",
        labeled_df_5["charge"][67],)

#%% Save in .pkl file.
# out_path="Detroit562/2016_02_27_output/DT_M2105_386_tensor_dataset.pkl"

labeled_df_5.to_pickle(out_path)
print(f"Saved in file: {out_path}")