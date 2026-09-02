#!/usr/bin/python3
"""
pyteomics-4.6 

requirement
-Two mandetory variable(两个变量):一个mgf,一个tsv. The file name must be same(文件名要匹配)
-One optional variable(另外一个变量): modification,must specify element compasition(要写出 氨基酸字母：元素构成（搞成配置文件）)
working directory(工作目录):/mnt/g/Lund_master_thesis/pyteomics

ToDo List:
    * 这个program一共有5个input变量,需要手动从外部传入:
        - mgf_folder_name    # "Detroit562/mgf_folder/"
        - mgf_file_name      # "DT_M2106_048.mgf"
        - csv_folder_name    # "Detroit562/"
        - csv_file_name      # "DT_M2106_048.csv"
        - output_folder_name
        
    * Output 的 path是自动生成的: f"{csv_folder_name}{name}_before_annotation.pkl"
    * 根目录也是自动生成的

把“文件名公共部分”当作唯一输入，比如 DT_M2105_386

    - id                 (外部传入) # "DT_M2106_048"
    - mgf_folder_name    (外部传入) # "Detroit562/mgf_folder/"
    - mgf_file_name         (自动生成)# "{id}.mgf"
    - csv_folder_name    (外部传入) # "Detroit562/"
    - csv_file_name         (自动生成) # "{id}.csv"
    - output_folder_name (外部传入) # "2016_02_27_Combine_mgf_and_percolater/"


Usage: 
    python script.py --base Detroit562 --id DT_M2105_386 --out output/
"""




#%% Import packages and check versions
import sys
import pyteomics
from pyteomics import mgf
import pandas as pd
import importlib.metadata
print("Package imported successfully.")
print(f"pyteomics: {importlib.metadata.version('pyteomics')}")
print(f"pandas: {pd.__version__}")
print(f"python: {sys.version.split()[0]}")

# a seting that print all columns out
pd.set_option('display.max_columns', None)
# "mgf" file includes first four columns
# "tsv" file includes "sequence"
# "lable" should be computed by this program

# Create a DataFrame
#column: Scan;retention_time;charge;m/z;sequence;lable
# Create a table
#%% 查看当前工作目录:/home/huimin
import os 
print(os.getcwd()) 
#%% 设置程序运行根目录,自动生成的
from pathlib import Path

# 获取当前script所在目录
current_dir = Path(__file__).resolve().parent
print("current_dir:",current_dir)       # current_dir: /home/huimin/Prepocessing

# 获取parent_dir 
print("parent_dir:",current_dir.parent) # parent_dir:  /home/huimin
parent_dir=current_dir.parent

# 设置program的根目录为 parent_dir
os.chdir(parent_dir)
print("New working directory:", Path.cwd())

#%% argv
import argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--id", required=True)
    p.add_argument("--mgf_folder_name", required=True)
    p.add_argument("--csv_folder_name", required=True)
    p.add_argument("--output_folder_name", required=True)
    return p.parse_args()

def build_paths(args):

    mgf_dir = Path(args.mgf_folder_name)
    print("mgf_dir:",mgf_dir)

    csv_dir = Path(args.csv_folder_name)
    print("csv_dir:",csv_dir)

    out_dir = Path(args.output_folder_name)
    print("out_dir:",out_dir)


    mgf_path = mgf_dir / f"{args.id}.mgf"
    print("mgf_path:",mgf_path)

    csv_path = csv_dir / f"{args.id}.csv"
    print("csv_path:",csv_path)

    id= f"{args.id}"
    print("id:",id)

    # 检查输入文件是否存在（建议保留）
    if not mgf_path.is_file():
        raise FileNotFoundError(f"MGF not found: {mgf_path}")
    #print(f"MGF found: {mgf_path}")

    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    #print(f"CSV found: {csv_path}")

    # 输出目录不存在就创建
    out_dir.mkdir(parents=True, exist_ok=True)

    return mgf_path, csv_path, out_dir, id

def main():
    args = parse_args()
    mgf_path, csv_path, out_dir, id  = build_paths(args)

    # 你的核心逻辑放这里：读 mgf + 读 csv + combine + save
    print("MGF:", mgf_path)
    print("CSV:", csv_path)
    print("OUT:", out_dir)
    print("ID:",  id)
    return mgf_path, csv_path, out_dir, id

if __name__ == "__main__":
    mgf_path, csv_path, out_dir, id= main()
    print("Finish")

#%% Input file folder: Detroit562/ 
###   mgf file folder: Detroit562/mgf_folder/
# mgf_folder_name="Detroit562/mgf_folder/"
# mgf_file_name="DT_M2106_048.mgf"

# 构建路径
# mgf_file='Detroit562/mgf_folder/DT_M2106_048.mgf'
# mgf_path= mgf_folder_name + mgf_file_name
print("mgf_file path:", mgf_path)
# /home/huimin/Detroit562/mgf_folder/DT_M2106_048.mgf
# if mgf_file == mgf_path :
#     print("yes,equal")

#%% Adding first four columns from "mgf" file
df = pd.DataFrame(columns=[
                            'ScanNr', 
                            'rtinseconds', 
                            'pepmass', 
                            'charge', 
                            'm/z',
                            'intensity'
                            ])

# spectrum["params"]: 'scans', 'pepmass', 'charge', 'rtinseconds'
# spectrum["m/z array"]: one array
# spectrum["intensity array"]: one array
i = 0
with pyteomics.mgf.read(str(mgf_path)) as reader:#"reader" is the whole mgf file.
    for spectrum in reader:# read spectra one by one
        # print(i)
        # print("param")
        scan_num=spectrum["params"]['scans']#scan number
        retention_time=spectrum['params']['rtinseconds']#retention time, type:<class 'pyteomics.auxiliary.structures.unitfloat'>
        retention_time=float(retention_time) #transforming into float type
        
        pepmass=spectrum['params']['pepmass'][0] # pepmass: <class 'float'>        
        #to get the charge
        if 'charge' in spectrum['params'].keys():
            charge=int(str(spectrum['params']['charge'][0])[:-1])#Charge
        else:
            charge=""
        # print(f"ScanNr:{scan_num}\t retension time:{retention_time}\t pepmass:{pepmass}\t charge:{charge}")
        # print("scan_num",type(scan_num))
        # print("retention time:",type(retention_time))
        # print("charge",type(charge))
        mzRatio=spectrum["m/z array"]#m/z Ratio
        # print(type(mzRatio))
        # print(f"mzRation:{mzRatio}")
        
        intensity=spectrum["intensity array"]
        # print(f"intensity:{intensity}")
        
        df.loc[i] = pd.Series({ 'ScanNr': scan_num,            # <class 'str'>
                                'rtinseconds': retention_time, # <class 'float'>
                                'pepmass':pepmass,             # <class 'float'>     
                                'charge':charge,               # <class 'int'>
                                'm/z': mzRatio,                # <class 'numpy.ndarray'>
                                'intensity': intensity,        # <class 'numpy.ndarray'>
                                } )
        i+=1
        
#%% "df", shape:  (84890, 6)
print(f"{id}.mgf Datafarme")
print(f"\t shape:  {df.shape}")
print(f"\t columns:{df.columns}") #['ScanNr', 'rtinseconds', 'pepmass', 'charge', 'm/z', 'intensity']
print(f"\t unique scan amount: {df['ScanNr'].nunique()}")


#%% csv file name
# csv_folder_name="Detroit562/"
# csv_file_name="DT_M2106_048.csv"

# csv_file="Detroit562/DT_M2106_048.csv"
# csv_path= csv_folder_name + csv_file_name

# if csv_file==csv_path:
#     print("Yes, equal")
# else:
#     print("No")

#%% open csv file (12477, 3)
print(f"Open file {id}.csv ")
percolator_df = pd.read_table(
                            csv_path,  # Input file, csv file
                            sep=',',   # delimiter is comma
                            )
# Three columns: 'TITLE', 'DenovoPeptideSequence', 'SpectrumScore'
print(f"\t shape:  {percolator_df.shape}")
print(f"\t columns:{percolator_df.columns}")
print(f"\t unique scan amount:{len(percolator_df.TITLE.unique())}")
#print(percolator_df.head())

#%% Filtering 
# print("Filtering: remove the sequences that are not end with K or R")
# filterring sequence and generating a column 'clean_seq'
# Trypsin 酶切规则：在 K 或 R 后切割
# 至少 2 个氨基酸
# 且最后一个必须是 K 或 R

# seq_pattern = r'^[A-Z]{2,}[KR]$'

# UniMod 是一个标准化的蛋白质修饰数据库
# 在质谱数据库搜索(如 Comet)中，UniMod:编号 表示某一个具体修饰。
# UniMod:35 = Oxidation, in Comet: variable_mod01 = 15.994915 M 0 3 -1 0 0

## 1, 创建 "clean_seq" column, 删除 (UniMod:xxx)
# percolator_df['clean_seq'] = (
#                                 percolator_df['DenovoPeptideSequence']          #取出这一列(一个 pandas Series)
#                                 .str.replace(r'\(UniMod:\d+\)', '', regex=True) #删除 (UniMod:xxx)
#                             )

## 2, 过滤
# percolator_df = percolator_df[
#                                 percolator_df['clean_seq'].str.match( seq_pattern ) #判断整个字符串是否满足K/R结尾
#                             ].reset_index(drop=True) #重新生成干净的连续index,不保留原始index

#%% After filtering

#print(df.head())
#print(percolator_df.head())

#print(df.columns)
#print(percolator_df.columns)

#print(df.shape)             #(84890, 6)
#print(percolator_df.shape)  #( 8486, 4)

print("After filtering")
print("The new csv Dataframe")
print(f"\t shape:  {percolator_df.shape}")
print(f"\t columns:{percolator_df.columns}")
print(f"\t unique scan amount:{len(percolator_df.TITLE.unique())}")



#%% merge info table "df" and post-filtering table "percolator_df". Generating a new table "merged_df"
print("Merging these two DataFrames and saved it")
merged_df = df.merge( percolator_df,
                      left_on='ScanNr', #用 df 的 ScanNr 作为 key
                      right_on='TITLE', #用 percolator_df 的 TITLE 作为 key
                      how='right'       #保留'percolator_df'所有行,按'TITLE'去"df"找匹配
                      ).drop(columns='TITLE') #不想保留 "TITLE"重复列

print(f"The output Dataframe of {id}")
print(f"\t shape: {merged_df.shape}") # (8486, 9)
print(f"\t shape: {merged_df.columns}")


#%% Save in pkl file
# output file name
# csv_file_name="DT_M2106_048.csv"
# csv_folder_name="Detroit562/"

# Save in pkl file
output_file_path= f"{out_dir}/{id}_before_annotation.pkl"
print(f"output_file_path is: {output_file_path}")
merged_df.to_pickle(output_file_path)
print(f"Saved in file: {output_file_path}")
print("###########################################################################################")