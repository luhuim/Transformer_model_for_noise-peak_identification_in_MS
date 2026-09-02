# /usr/bin/python3
"""
Update: 
    这个program算theoretical fragment ions的时候要用到的ion types和charge如下:
        - ion_types=['a','b','y']
        - charge_list=[ 1 , 2 ]
    program name:/home/huimin/Prepocessing/2026_04_07_calculate_ion_mass.py

Introduction:为每条 peptide 构建完整的 theoretical fragment ion m/z 空间（考虑修饰 + 电荷 + 离子类型）
This part is modification 
working directory(工作目录):/mnt/g/Lund_master_thesis/pyteomics

Input file:
    'DT_M2105_468_before_annotation.tsv'
Output files:
    spectrum charge is "NaN": "DT_M2105_468_null_charge.tsv"
    output file used in following steps: 'DT_M2105_468_before_comparing.tsv'

Requirement:
    pandas higher than 2.0.0
    pyteomics
    numpy
    re

    - config_name
    - pkl_dir
    - pkl_name
    - out_dir ="Detroit562/"
    - out_name="DT_M2106_048_before_comparing.pkl"        
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

#%% 查看当前工作目录:/home/huimin
import os 
print(os.getcwd()) 

#%% 设置程序运行根目录,自动生成的
from pathlib import Path

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
From argv:
    - config_name
    - id
    - pkl_dir ="2016_02_27_output"
    - out_dir ="2016_02_27_output"

Generate
    - pkl_name="{id}_before_annotation.pkl" # in
    - out_name="{id}_before_comparing.pkl"  # out
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
    pkl_path = pkl_dir / f"{id}_before_annotation.pkl"
    print("pkl_path:",pkl_path)
    #output path
    out_path = out_dir / f"{id}_before_comparing.pkl"
    print("out_path:",out_path)

    # 检查input (.pkl)文件是否存在(建议保留)
    if not pkl_path.is_file():
        raise FileNotFoundError(f"Input file not found: {pkl_path}")
    print(f"Input (.pkl) found: {pkl_path}")

    # 输出目录不存在就创建
    out_dir.mkdir(parents=True, exist_ok=True)

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
default = [ 'a','b','y']
# Define config 字典
config = {}

# 判断config文件是否存在
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {} #config is a dictionary
        # print(config)

# 获取 "ion_types"
ion_types = config.get( # 从config字典当中获取key,"ion_types",
                        "ion_types", 
                        default, #如果config当中没有这个key,就用默认值
                        )

print(f"ion_types:{ion_types}")

#%% 构建input pkl路径
# pkl_dir = "Preprocessing_output/"
# pkl_name= "DT_M2106_039_before_annotation.pkl" 

# 构建路径
# pkl_path= pkl_dir + pkl_name

#%% Open the input file:"DT_M2106_039_before_annotation.pkl"
print(f"Open file:{pkl_path}") 
# Open the input file (.pkl)
before_annotation = pd.read_pickle(pkl_path)
# the shape of DataFrame
print("DataFrame shape:",before_annotation.shape)

#%% Observing the DataFrame
print(f"before_annotation shape:{before_annotation.shape}")
print(f"columns: {before_annotation.columns}")
# ['ScanNr', 'rtinseconds', 'pepmass', 'charge', 'm/z', 'intensity','DenovoPeptideSequence', 'SpectrumScore', 'clean_seq']

#%% 提取 peptide sequence 放到 "clean_seq" column; 创建 empty "ion_mass" columns

# "clean_seq" col 删除 (UniMod:xxx) 
before_annotation['clean_seq'] = (
                before_annotation['DenovoPeptideSequence']          #取出这一列(一个 pandas Series)
                .str.replace(r'\(UniMod:\d+\)', '', regex=True)     #删除 (UniMod:xxx)
                             )
# print(before_annotation['DenovoPeptideSequence'])
# print(before_annotation['clean_seq'])

# Adding an empty column:"ion_mass"
before_annotation["ion_mass"] = np.nan
before_annotation["ion_mass"] = before_annotation["ion_mass"].astype('object')

#%% Calculating theoretical ion mass, and save the new array in column ""ion mass""
from pyteomics import mass
# UniMod dictionary
db = mass.Unimod()
# 定义 regex pattern. 这个pattern用来匹配: 氨基酸+UniMod修饰
pattern = r'([A-Z])(\(UniMod:\d+\))'
# 
ion_types=['a','b','y']
#%% Define a custemized "aa_comp" dict
unimod_dic= { "M":35,
              "C":4,}

# a user-defined aa_comp dict
aa_comp = dict(mass.std_aa_comp) # Original dictionary

# add information about modified amino acids to a user-defined aa_comp dict
db = mass.Unimod()

# M(UniMod:35)
mod_M = db.by_id(35) # 提取这个'UniMod_ID'的信息, "mod_M"
# 把 M 重定义为：原始 M + UniMod35
aa_comp["M"] = mass.std_aa_comp["M"] + mod_M['composition']

# C(UniMod:4)
mod_C = db.by_id(4) # 提取这个'UniMod_ID'的信息, "mod_C"
# 把 C 重定义为：原始 C + UniMod4
aa_comp["C"] = mass.std_aa_comp["C"] + mod_C['composition']

#%% Calculate theoretical ion mass and label 
ion_types=['a','b','y']
charge_list=[ 1 , 2 ]
# read the dataFrame one by one 👉 DataFrame 每一行 = 一个 spectrum
for i in range ( 0, len(before_annotation)):
    print("i=",i)
    theoretical_mass=[]# create a list, to save ion mass,called "theoretical_mass"

    # Getting one spectrum: ScanNr = DT_M2106_039.31045.31045.3, index = 6201
    # one_spectrum = before_annotation[ before_annotation["ScanNr"] == "DT_M2106_039.31045.31045.3"]
    # print(one_spectrum.index) # index= 6201
    
    # Getting the spectrum: 'm/z','intensity','charge','DenovoPeptideSequence','clean_seq'
    mz_ratio = before_annotation.loc[i,"m/z"]
    intensity= before_annotation.loc[i,"intensity"]
    charge=before_annotation.loc[i,"charge"]
    # print(f"mz_ratio:{mz_ratio}")
    # print(f"intensity:{intensity}")
    # print(f"charge:{charge}")
    full_seq = before_annotation.loc[i,"DenovoPeptideSequence"]
    peptide = before_annotation.loc[ i,'clean_seq']
    # print(f"full_seq:{full_seq}")
    # print(f"peptide:{peptide}")

##### Theoretical peptide mass: (percurser)
    # Neutral Mass,or Monoisotopic Mass (M): monoisotopic=用最轻同位素计算, neutral = 不带电
    neutral_mass = mass.calculate_mass( sequence=peptide, aa_comp=aa_comp,)
    print(f"neutral_mass (M): {neutral_mass}")
    # Protonated Mass, or [M+H]+ m/z: 整个 peptide 加一个质子的离子质量
    protonated_mz = mass.calculate_mass(
                                sequence=peptide, 
                                ion_type='M',    # 'M'完整分子离子(molecular ion / precursor ion)
                                charge=1, 
                                aa_comp=aa_comp,      
                                )
    print(f"protonated mass [M+H]+: {protonated_mz}")

    protonated_2H = mass.calculate_mass(
                                sequence=peptide, 
                                ion_type='M',    # 'M'完整分子离子(molecular ion / precursor ion)
                                charge=2, 
                                aa_comp=aa_comp,      
                                )
    print(f"[M+2H]2+: {protonated_2H}")
 
    
    # [M+H−H2O]+ :  单电荷(z=1) + 脱水, 
    H2O = mass.calculate_mass(formula="H2O")
    loss_1_water_1_charge = protonated_mz-H2O
    print(f"[M+H-H2O]+:{loss_1_water_1_charge}")

    # [M+2H−H2​O]2+ : 双电荷(z=2) + 脱水
    M = mass.calculate_mass( sequence=peptide, aa_comp=aa_comp )
    proton = mass.calculate_mass(formula="H")
    H2O = mass.calculate_mass(formula="H2O")
    loss_1_water_2_charge = (M + 2*proton - H2O) / 2
    print(f"[M+2H-H2O]2+: {loss_1_water_2_charge}")


    # [M+H−NH3​]+
    NH3 = mass.calculate_mass(formula="NH3")
    loss_1_NH3_1_charge = protonated_mz - NH3    
    print(f"[M+H-NH3]+: {loss_1_NH3_1_charge}")
    
    #[M+2H−NH3​]2+
    M = mass.calculate_mass(sequence=peptide, aa_comp=aa_comp)
    loss_1_NH3_2_charge = (M + 2 * proton - NH3)/2
    print(f"[M+2H-NH3]2+: {loss_1_NH3_2_charge}")

    print(f"Peptide: {peptide}")
    print(f"Neutral_mass (M): {neutral_mass}")
    print(f"Protonated mass  [M+H]+: {protonated_mz}, [M+2H]2+: {protonated_2H}")
    print(f"Neutral loss H20 [M+H-H2O]+:{loss_1_water_1_charge},[M+2H-H2O]2+: {loss_1_water_2_charge}")
    print(f"Neutral loss NH3 [M+H-NH3]+:{loss_1_NH3_1_charge},  [M+2H-NH3]2+: {loss_1_NH3_2_charge}")
    theoretical_mass.extend([neutral_mass,              # [M]
                             protonated_mz,             # [M+H]+
                             protonated_2H,             # [M+2H]2+
                             loss_1_water_1_charge,     # [M+H-H2O]+
                             loss_1_water_2_charge,     # [M+2H-H2O]2+
                             loss_1_NH3_1_charge,       # [M+H-NH3]+
                             loss_1_NH3_2_charge,       # [M+2H-NH3]2+
                            ])


##### Theoretical ion mass: a-ion, b-ion, y-ion, charge = 1, 2
    print("切碎片的loop: 计算ion mass") 
    print("-------")
    
    for c in charge_list: # charge = 1, 2
        print(f"Charge={c}")
        for s in range(1,len(peptide)): # cut peptide into pieces,called "seq" 切碎片 
            ######### Calculate a-ion and b-ion
            seq = peptide[:s]       # "seq" is one piece of peptide 碎片
            # print(f"Front_seq:{seq}")   # "seq" is 碎片    
            a = mass.calculate_mass( # "a" is mass of this ion
                                    sequence=seq,       # 碎片 front
                                    ion_type='a',       # 离子类型 a-ion
                                    charge=c,           # 电荷 c
                                    aa_comp=aa_comp,    # 使用 User-defined 'aa_comp' dict
                                    ) 
            # print(f"a-ion,charge=1, mass={a}")
            theoretical_mass.append(a)

            b = mass.calculate_mass( 
                                    sequence=seq,       # 碎片 front
                                    ion_type='b',       # 离子类型 b-ion
                                    charge=c,           # 电荷 c
                                    aa_comp=aa_comp,    # 使用 User-defined 'aa_comp' dict
                                    ) 
            # print(f"b-ion,charge=1, mass={b}")
            theoretical_mass.append(b)

            ######### Calculate y-ion
            rear_seq = peptide[s:]
            # print(f"Rear_seq:{rear_seq}")
            y = mass.calculate_mass( 
                                    sequence = rear_seq,  # 碎片 rear
                                    ion_type = 'y',       # 离子类型 y-ion
                                    charge=c,             # 电荷 c
                                    aa_comp=aa_comp,      # 使用 User-defined 'aa_comp' dict
                                    )
            # print(f"y-ion,charge=1, mass={y}") 
            theoretical_mass.append(y)

            print(f"Front_seq:{seq},a-ion={a},b-ion={b}; Rear_seq:{rear_seq}, y-ion={y}.")
    # 切碎片的loop END
    # print(theoretical_mass)
    theoretical_mass =  np.array(theoretical_mass) # Transform list into numpy array
    
    # Insert array into DataFrame
    before_annotation.at[before_annotation.index[i], "ion_mass"] = theoretical_mass
                      # "before_annotation.index[i]"👉 做了这件事："第i行" → 对应的 index label

    #before_annotation["ion_mass"][i]=m  #(旧版)这种DataFrame赋值方式可能触发："SettingWithCopyWarning"
    print("One spectra finished.--------------")
print("All spectrum finished")
print(before_annotation.columns)
print(before_annotation.shape)
#%% Save in pkl file
# out_dir ="Preprocessing_output/preprocessing_output_2026_05_03"
# out_name="DT_M2106_039_before_annotation.pkl" 
# 构建路径
# out_path= out_dir + out_name

# Save in pkl file
before_annotation.to_pickle(out_path)
print(f"Saved in file: {out_path}")
print("###########################################################################################")

#%% re.finditer( pattern, seq)
# import re
# from pyteomics import mass

# db = mass.Unimod()
# # A sequence from column "DenovoPeptideSequence"
# seq = "VKM(UniMod:35)LPDKDC(UniMod:4)RY"
# # 定义 regex pattern. 这个pattern用来匹配: 氨基酸+UniMod修饰
# pattern = r'([A-Z])(\(UniMod:\d+\))'

#  [A-Z] :   表示匹配：任意一个大写字母
# ([A-Z]):   括号表示：capture group, 也就是说：把匹配到的内容保存下来,这里保存的是氨基酸

# \(UniMod:\d+\) 这部分匹配： (UniMod:35) (UniMod:4)
# \(        匹配 "("
# UniMod:   固定字符串
# \d+       一个或多个数字
# \)        匹配 ")"

#第二个括号 (\(UniMod:\d+\)) 也是 capture group. 保存： (UniMod:35) (UniMod:4)
# group1 → M
# group2 → (UniMod:35)

# 这是一个 peptide sequence, 其中包含修饰:  V K M(UniMod:35) L P D K D C(UniMod:4) R Y
#意思是: 第 3 个氨基酸 M 有 Oxidation
#       第 9 个氨基酸 C 有 Carbamidomethyl
#[('M', 3, '(UniMod:35)'), ('C', 12, '(UniMod:4)')]

# mod_dict={}
# for match in re.finditer( pattern, seq):
#     print(f"match:{match}")
#     # 
#     UniMod_id = int(match.group(2)[1:-1].split(":")[1]) # (UniMod:xx)
#     # 
#     aa =  match.group(1)                # amino acid
#     index = match.start(1)              # amino acid index
#     # Checking the value and type
#     print(f"UniMod_id:{UniMod_id}")
#     print(f"aa:{aa};index:{index}")
#     print(type(aa),         # <class 'str'>
#           type(index),      # <class 'int'> 
#           type(UniMod_id),) # <class 'int'>
    
#     # Adding into dictionary
#     mod_dict[index] = {}
#     mod_dict[index]["aa"] = aa 
#     mod_dict[index]["UniMod_id"] = UniMod_id
# print(mod_dict.keys())
# print(mod_dict[2]["aa"],mod_dict[2]["UniMod_id"])

# unimod_id=mod_dict[2]["UniMod_id"]
# print(db.by_id(unimod_id)["composition"])
# print(db.by_id(unimod_id)['mono_mass'])

# {index:
#         {"aa":aa,
#         "UniMod_id":UniMod_id}
# }



#%% mass.Unimod()

#db = mass.Unimod()
#print(db.by_id(4)["composition"])
#print(db.by_id(4)['mono_mass'])


# db.by_id(4)
#{
# 'date_time_modified': datetime.datetime(2017, 10, 9, 10, 27, 10),  #数据库记录修改时间
# 'date_time_posted': datetime.datetime(2002, 8, 19, 19, 17, 11),    #数据库记录创建时间
# 'avge_mass': 57.0513,   #平均质量(average mass)在高分辨率质谱中通常不用
# 'mono_mass': 57.021464, # 单同位素质量(monoisotopic mass)这是质谱计算时真正使用的质量
# 'composition': Composition({'H': 3, 'C': 2, 'N': 1, 'O': 1}),      #元素组成
# 'record_id': 4, #这是 Unimod 数据库里的唯一ID
# 'approved': True,
# 'title': 'Carbamidomethyl',               #修饰名字
# 'full_name': 'Iodoacetamide derivative',  #修饰的完整描述
# 'username_of_poster': 'unimod',           #数据库提交者
# 'group_of_poster': 'admin',
# 'specificity': [ # 这个修饰可以发生在哪些 amino acid上
#           {'hidden': True, 'spec_group': 9, 'site': 'Y', 'position': 'Anywhere', 'classification': 'Artefact'},
#           {'hidden': True, 'spec_group': 8, 'site': 'T', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': True, 'spec_group': 7, 'site': 'S', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': True, 'spec_group': 6, 'site': 'E', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': True, 'spec_group': 5, 'site': 'D', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': True, 'spec_group': 4, 'site': 'H', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': False,'spec_group': 3, 'site': 'N-term', 'position': 'Any N-term', 'classification': 'Artefact'}, 
#           {'hidden': True, 'spec_group': 2, 'site': 'K', 'position': 'Anywhere', 'classification': 'Artefact'}, 
#           {'hidden': False,'spec_group': 1, 'site': 'C', 'position': 'Anywhere', 'classification': 'Chemical derivative'}, 
#           {'hidden': True, 'spec_group': 10,'site': 'U', 'position': 'Anywhere', 'classification': 'Chemical derivative'}, 
#           {'hidden': True, 'spec_group': 11,'site': 'M', 'position': 'Anywhere', 'classification': 'Chemical derivative'}
#               ], 
# 'alt_names': ['Carboxyamidomethylation'], #修饰的别名
# 'refs': [ #refs(文献)
#          {'text': '11510821', 'source': 'PubMed PMID', 'url': None}, 
#          {'text': '12422359', 'source': 'PubMed PMID', 'url': None}, 
#          {'text': 'Boja, E. S., Fales, H. M., Anal. Chem. 73 3576-82 (2001)', 'source': 'Journal', 'url': None}, 
#          {'text': 'Creasy, D. M., Cottrell, J. S., Proteomics 2 1426-34 (2002)', 'source': 'Journal', 'url': None}
#         ]
#}





#%% pyteomics unimod from www.unimod.org
# 1: Accession:    这一列就是 UniMod ID
# 2: PSI-MS Name:  这是HUPO PSI-MS Ontology 中使用的标准名称
#    PSI-MS是 Human Proteome Organization制定的质谱术语标准
# 3: Interim name:  这是临时名称,很多常见修饰这里是空的
# 4: Description:   修饰的文字说明. 例如：Oxidation 通常描述为"Addition of one oxygen atom"
# 5: Monoisotopic mass: 单同位素质量变化(mass shift),质谱计算通常使用这个值
# 6: Average mass:平均质量变化.计算方式使用自然同位素丰度.高分辨率质谱中几乎总是使用 monoisotopic mass
# 7: 7 Composition: 这是修饰的化学元素组成

# UniMod_shift_mass={
#                 1:42.010565,
#                 4:57.021464,
#                 5:43.005814,
#                 7:0.984016,
#                 27:-18.010565,
#                 28:-17.026549,
#                 34:14.015650,
#                 35:15.994915,
#                 385:-17.026549,
#             }
