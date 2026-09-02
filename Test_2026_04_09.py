#/usr/bin/python3
"""
Test.py

Input: 
    checkpoint:
    test-set:
Output:

"""
#%% Import packages 
import torch
from torch import nn
import torch.optim as optim

import pandas as pd
import numpy as np
import math

from dataclasses import dataclass, field
from typing import Dict, Optional

#系统文件操控的module
import os
from pathlib import Path
import yaml # install command: conda install pyyaml
print("Done: import packages")

#%% YAML file name
config_path = "config_2026_06_26.yaml"

#%% Check GPU
from torch.cuda import amp
#--- 设备选择 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Device] Using: {device}")

#Print the GPU and cuda infomation 
print("Version of torch:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("current_device:", torch.cuda.current_device())
    print("name:", torch.cuda.get_device_name(0))
    
#Command of getting GPU info
#!nvidia-smi
#This is job ID
print(os.getpid())

#%% YAML file, weight tensor

default_weight = [1.0, 3.8]

if os.path.exists(config_path):
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        if (
            isinstance(config, dict)
            and "class_weight" in config
            and isinstance(config["class_weight"], (list, tuple))
            and len(config["class_weight"]) == 2
        ):
            weight_values = config["class_weight"]
            print("Config file exists")
            print(f"Using class_weight from config file: {weight_values}")
            weight = torch.tensor( weight_values, dtype=torch.float32).to(device, non_blocking=True)
        else:
            print("config.yaml found but class_weight invalid, using default.")
            weight = torch.tensor([1.0, 3.8]).to(device, non_blocking=True)
    except Exception as e:
        print(f"Error reading config.yaml: {e}")
        print("Using default class_weight.")
        weight_values = default_weight
else:
    print("config.yaml not found, using default class_weight.")
    weight = torch.tensor([1.0, 3.8]).to(device, non_blocking=True)

print(f"weight is:{weight}")

#%% Create output folder
from pathlib import Path

# 当前 Python 文件的完整路径
current_file = Path(__file__).resolve()

# 当前 Python 文件所在的文件夹
current_dir = current_file.parent

# Python 文件名（不含 .py）
folder_name = current_file.stem

# 创建与 Python 文件同名的文件夹
output_dir = current_dir / folder_name
output_dir.mkdir(exist_ok=True)

print(f"Folder created at: {output_dir}")

print(type(output_dir),output_dir)

#%% 打开二进制文件(open '.pkl' file)加载'loaded_df'
import pickle
print("Open test_set .pkl file and load DataFrame.")
# 2026_04_10_test_dataset.pkl
with open("2026_04_10_test_dataset.pkl",'rb') as f:
    loaded_df = pickle.load(f)
print("test_set has been loaded from pkl file.")
print(f"The shape of test-set: {loaded_df.shape}") # 24,897

# Column name
print(f"Columns:{loaded_df.columns}")
#%% making pytorch dataset
# 在创建Dataset之前，从train_set DataFrame里过滤掉peak数量>200的spectrum
MAX_PEAKS = 200
loaded_df = loaded_df[
                    loaded_df['filtered_mz'].apply(lambda x: len(x.squeeze(0)) <= MAX_PEAKS)
                    ].reset_index(drop=True)
print(f"Peak < 200, amount{len(loaded_df)}") # 24,515

# 从train_set DataFrame里过滤掉label全是零的spectrum
loaded_df = loaded_df[  loaded_df['filtered_ration_of_1s'] != 0].reset_index(drop=True)
print(f"Peak < 200; 过滤掉label全是零, amount{len(loaded_df)} ")# 24,513 

# print 5 rows
# loaded_df.head()

print("Done\n")
#%% 创建test_dataloader的实例
from Dataloader.Dataloader_24_06_24 import SpectrumDataset_new, collate_fn, PeakLengthSortedSampler # Dataset的合成调用了外部的class
from torch.utils.data import DataLoader # DataLoader的合成用的是torch自带的功能
print("Build test_dataloader.")
# 创建train Dataset实例:把'loaded_df'创建test_dataset实例
test_dataset = SpectrumDataset_new(loaded_df)

#创建按 peak 数量排序的 Sampler
test_sampler = PeakLengthSortedSampler(
                                        test_dataset,
                                        descending=True # False: 少→多；True: 多→少
                                        )

#创建 DataLoader, 使用自定义的collate_fn 
test_dataloader = DataLoader(
                                test_dataset,  #这里不能动
                                batch_size=16,  #这里可以改 #Volley
                                shuffle= False,       #valid-set要保持顺序False
                                collate_fn = collate_fn,
                                # sampler=train_sampler,  # 使用 sampler 控制顺序
                                drop_last=False, # 默认不丢弃最后一个不足一个batch_size的数据
                    			num_workers=0,    	      # new
                        		# prefetch_factor=2,	  # new
                        		persistent_workers=False, # new
                        		pin_memory=False,         # new 
        			        )
print("Done\n")

#%% 部署model(1/3): 实例化model ============================================================================
print("Building the model architecture.")
import torch.optim as optim
# 导入的class和之前不一样.Path: ~/pyteomics/TransformerBlock/PositionalEncoding_24_07_02.py
# from TransformerBlock.PositionalEncoding_24_07_02 import PeakEncoder
from TransformerBlock.Encoder_24_07_05 import SpectrumTransformerEncoder
# from TransformerBlock.Encoder_24_07_05 import MassEncoder #编码pepmass
from TransformerBlock.Decoder_25_12_08 import TransformerDecoderCaseB, PositionalEncoding

d_model = 512
lr = 1e-3, # learning rate
max_charge=10
# optimizer = optim.Adam( lr=lr)
# 设置超参数
L_peaks = 10    # 每个样本的谱峰个数
d_model = 512
nhead = 8
num_layers = 2
dim_feedforward = 64

#实例化
# 编码/嵌入spectrum信息 
#1.创建PeakEncoder实例 positional encoding质谱的peaks
# peak_encoder = PeakEncoder(d_model).to(device)  
#2.创建MassEncoder示例 embedding pepmass (float)
# mass_encoder = MassEncoder(dim_model=d_model).to(device) 
#3.创建ChargeEncoder示例 embedding charge (int)
# charge_encoder = nn.Embedding(max_charge, d_model).to(device) 

#=================================================
#4.创建encoder实例(含 positional encoder)
encoder = SpectrumTransformerEncoder(
                                    d_model = d_model, 
                                    dim_feedforward = 1024, 
                                    n_layers = 2, 
                                    dropout = 0.1,
                                    # peak_encoder=peak_encoder,
                                    # mass_encoder=mass_encoder,
                                    ).to(device) 

#5.构建 decoder
decoder = TransformerDecoderCaseB(
                                    d_model=d_model,
                                    nhead=nhead,
                                    num_layers=num_layers,
                                    dim_feedforward=dim_feedforward,
                                    dropout=0.1,
                                    bos_token=2,
                                    ).to(device)

print("Done\n")
#%% 部署model(2/3): 读取checkpoint(.pth文件) 
print("Loading checkpoint file.")
#Full path of "CheckPoint file: MasterThesis/RunOnServer/TrainingModel_2025_11_03_CheckPoint/checkpoint_epoch5.pth
PATH="./TrainingModel_2026_06_26/checkpoint_epoch24.pth" # the path of checkpoint file

checkpoint = torch.load(PATH, map_location=device) # Import .pth file

print("Keys in checkpoint:", checkpoint.keys())
print("Checkpoint epoch:", checkpoint["epoch"])
print("Checkpoint loss:", checkpoint["loss"])

print("Done")

# checkpoint字典的结构如下
# """
# checkpoint={
#     'epoch':epoch,
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict':optimizer.state_dict(),
#     'loss': epoch_loss_sum,
#     }
# """

#%%部署model(3/3): 把checkpoint字典带入model和 optimizer
#%% 部署model(3/3): 将旧模型权重载入model achitecture
print("Loading weight into encoder and decoder.")
encoder.load_state_dict(checkpoint["encoder"])
decoder.load_state_dict(checkpoint["decoder"])

print("Encoder and decoder weights loaded successfully.")

#%% 建立好 encoder 和 decoder object 后统计 the number of parameter 
encoder_params = sum(p.numel() for p in encoder.parameters())
decoder_params = sum(p.numel() for p in decoder.parameters())

total_params = encoder_params + decoder_params

print(f"Encoder parameters: {encoder_params:,}")
print(f"Decoder parameters: {decoder_params:,}")
print(f"Total parameters: {total_params:,}")

#%% The number of trainable parameter

encoder_trainable = sum(
    p.numel() for p in encoder.parameters()
    if p.requires_grad
)

decoder_trainable = sum(
    p.numel() for p in decoder.parameters()
    if p.requires_grad
)

print(f"Trainable parameters: {encoder_trainable + decoder_trainable:,}")

#%% parameter breakdown: layer-wise parameter count (1/2)
encoder_sd = checkpoint["encoder"]
decoder_sd = checkpoint["decoder"]

def print_parameter_details(state_dict, module_name):
    print(f"\n{'=' * 100}")
    print(f"{module_name}")
    print(f"{'=' * 100}")

    total = 0

    for name, tensor in state_dict.items():
        n_params = tensor.numel()
        total += n_params

        print(
            f"{name:65s} "
            f"{str(tuple(tensor.shape)):25s} "
            f"{n_params:>12,}"
        )

    print("-" * 100)
    print(f"{module_name} total: {total:,} ({total / 1e6:.3f} M)")

    return total


encoder_params = print_parameter_details(
    encoder_sd,
    "ENCODER"
)

decoder_params = print_parameter_details(
    decoder_sd,
    "DECODER"
)

total_params = encoder_params + decoder_params

print(f"\n{'=' * 100}")
print(f"TOTAL MODEL PARAMETERS: {total_params:,} ({total_params / 1e6:.3f} M)")
print(f"{'=' * 100}")

#%% parameter breakdown: layer-wise parameter count (2/2)
from collections import defaultdict
import re

def count_transformer_layers(state_dict):
    layer_params = defaultdict(int)

    for name, tensor in state_dict.items():

        match = re.search(r'layers\.(\d+)', name)

        if match:
            layer_id = int(match.group(1))
            layer_params[layer_id] += tensor.numel()

    return layer_params


encoder_layers = count_transformer_layers(encoder_sd)
decoder_layers = count_transformer_layers(decoder_sd)

print("\nEncoder Transformer layers:")
for layer_id, n in sorted(encoder_layers.items()):
    print(
        f"Layer {layer_id}: "
        f"{n:,} parameters "
        f"({n / 1e6:.3f} M)"
    )


print("\nDecoder Transformer layers:")
for layer_id, n in sorted(decoder_layers.items()):
    print(
        f"Layer {layer_id}: "
        f"{n:,} parameters "
        f"({n / 1e6:.3f} M)"
    )

#%% trainable parameter  (1/2)
def print_model_parameters(model, model_name):
    print(f"\n{'=' * 100}")
    print(f"{model_name}")
    print(f"{'=' * 100}")

    total_params = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            n_params = param.numel()
            total_params += n_params

            print(
                f"{name:<70} "
                f"{str(tuple(param.shape)):<20} "
                f"{n_params:>12,}"
            )

    print("-" * 100)
    print(f"{model_name} trainable parameters: {total_params:,}")
    print(f"{model_name} trainable parameters: {total_params / 1e6:.3f} M")


print_model_parameters(encoder, "ENCODER")
print_model_parameters(decoder, "DECODER")

#%% Trainable parameter (2/2)
def print_parameter_summary(model, model_name):
    print(f"\n===== {model_name} =====")

    total = 0

    for name, module in model.named_modules():

        # recurse=False:
        # 只统计当前 module 自己的参数，
        # 不重复统计其 child modules
        params = sum(
            p.numel()
            for p in module.parameters(recurse=False)
            if p.requires_grad
        )

        if params > 0:
            total += params

            module_name = name if name else "[root]"

            print(
                f"{module_name:<70} "
                f"{params:>12,}"
            )

    print("-" * 85)
    print(f"Total: {total:,} ({total / 1e6:.3f} M)")


print_parameter_summary(encoder, "ENCODER")
print_parameter_summary(decoder, "DECODER")



#%% Parameter
epochs=20
log_interval= 1

#epoch iteration of model training 
epoch_train_loss_list=[] #This list saves loss of every epoch
all_epoch=[] # This list saves all DataFrame {precision, recall} of every epoch 
lr_list=[]   # This list saves learning rate of every epoch

#%% Test step
import matplotlib.pyplot as plt
import seaborn as sns
  
print("Move to test step")
# Validation mode, 此处'.eval()'的作用是:关闭 dropout、batchnorm 的训练行为
encoder.eval() 
decoder.eval()
print("validation mode")
# 清零 every epoch
epoch_dot=[]        # 本次epoch的,每个spectrum的,混淆矩阵 and metrics. one dataframe, one spectrum

with torch.no_grad(): #不计算梯度
    for i, batch in enumerate(test_dataloader):# validation_dataloader
        print(f"Validation batch {i}:") # batch的index
        # Step 2: 把Batch里面的input tensor搬到device
        spectrum = batch['spectrum'].to(device, non_blocking=True)      # [B, L, 2]
        print(
            f"spectrum shape: {spectrum.shape}, "
            f"dtype: {spectrum.dtype}, "
            f"device: {spectrum.device}"
            )

        target_seq = batch['target_seq'].to(device, non_blocking=True)  # [B, L]
        #print(
        #    f"target_seq shape:{target_seq.shape},"
        #   f"dtype: {target_seq.dtype},"
        #  f"device:{target_seq.device}"
        # )
        charge = batch['charge'].to(device, non_blocking=True)
        # print(
        #     f"charge shape:{charge.shape},"
        #     f"dtype: {charge.dtype},"
        #     f"device:{charge.device}"
        #         )
        pepmass = batch['pepmass'].to(device, non_blocking=True)
        # print(
        #     f"pepmass shape:{pepmass.shape},"
        #     f"dtype: {pepmass.dtype},"
        #     f"device:{pepmass.device}"
        #         )
        #################################(I)Encoder######################################################
        encoded_output, padding_mask = encoder(
                                        mz_array = spectrum[:, :, 0], #把spectrum的两层拆开分别带入encoder #torch.Size([3, 43, 1])                              
                                        intensity_array = spectrum[:, :, 1],#torch.Size([3, 43, 1])
                                        pepmass = pepmass,
                                        charge = charge,
                                        mask=None,
                                        )
        print("encoded_output shape:",encoded_output.shape)# torch.Size([3, 44, 512])
        print("padding_mask shape:",padding_mask.shape)# torch.Size([3, 44])
        # print("encoded_output:\n",encoded_output)
        # print("padding_mask:\n",padding_mask)

        ##################Decoder#########################################################
        # What is 'target_seq' for, in the validation loop?
            # They nees "target_seq" from second index to final as groun truth. both array and mask 
            # This tensor will not be fed into genenrater,
            # The generater() need tensor for encoder, and start token. 
                #the tensor if encoder has been added, shape: (batch_size,length,512) 
                #the start token can be created inside the generator(): 
                        # just obtain the batch_size, shape (batch_size,1) 
                        # fill this tensor with 2
        #准备正确答案
        # print(f"unique element of target_seq:{torch.unique(target_seq)}") # 1, 0,-1
        ground_truth  = target_seq[:, 1:]    # [B, L_peaks],提取第2位到最后1位, 用来算 loss 
        #print(f"正确答案:\n {ground_truth}")# include 0, 1 and -1, not include fixed 1.
        print(f"ground_truth shape: {ground_truth.shape}")
        # print(f"unique element of ground_turth: {torch.unique(ground_truth)}")
        #准备正确答案mask: False is token; True is padding
        ground_truth_mask=padding_mask[:, 1:]
        # print("正确答案mask 'ground_truth_mask' tensor:\n",tgt_target_mask)
        print(f"ground_truth_mask shape: {ground_truth_mask.shape}")
        print(f"False is token; True is padding:\n"
                f"\t the sum of False==token is {(ground_truth_mask == False).sum()};\n"
                f"\t the sum of True ==padding  is {(ground_truth_mask == True).sum()}\n")
        # Feed into decoder. The output is the seqeunce that it predicted 
        print("Before the decoder.generator. ")
        #打印一个参数验证
        print("打印一个参数验证确认是否为当前权重")
        print(f"'generate'之前:\n{next(decoder.parameters())[0][0:5]}")

        generated = decoder.generate(
                                    encoded_output,                     # from encoder
                                    src_key_padding_mask=padding_mask,  # from encoder
                                    # max_len=L_peaks,                    # 
                                    # greedy=False
                                    greedy=True,
                                )
        # logit1 = self.output_layer(last_h).squeeze(-1)  # (B,)
        # p1 = torch.sigmoid(logit1)                      # (B,)
        # next_token = (p1 > threshold).long().unsqueeze(1)  # (B,1)
        print("After the decoder.generator.")
        print(f"'generate'之后:\n{next(decoder.parameters())[0][0:5]}")
        # print("unique element if generated:",torch.unique(generated)) # it include: 0, 1
        print("Post-Processing: for-loop")
        ########
        # ground_truth shape: torch.Size([10, 88]) # include 0, 1 and -1, not include fixed 1.
        # ground_truth_mask : torch.Size([10, 88]) # False is token; True is padding.
        # generated         : torch.Size([10, 88]) # output tensor of decoder, including 0 and 1. 

        batch_size = generated.size(0)
        # 对答案 one by one, added into list "epoch_dot".
        for b in range( batch_size ):   # This 'for-loop' will go through every spectrum from this batch.
            print(f"spectrum:{b}")
            # Two types of sequence
            output_seq =  generated[b]         # output sequence from decoder, including 0 and 1.
            ground_truth_seq = ground_truth[b] # ground truth sequence include 0, 1 and -1, not include fixed 1.
            # One mask
            mask = ground_truth_mask[b]        # False is token; True is padding.

            print(f"output_seq:\n{output_seq}")
            print(f"ground_truth_seq:\n{ground_truth_seq}")
            print(f"mask:\n{mask}")

            print("Let these two sequences go through the mask.")
            output_seq_array=output_seq[~mask]
            ground_truth_seq_array=ground_truth_seq[~mask]

            print(f"output_seq_array:\n{output_seq_array}")
            print(f"ground_truth_seq_array:\n{ground_truth_seq_array}")
            
            if output_seq_array.sum() == 0:
                print("This sequence is full zero.")
                print(f"The ground truth array is:{ground_truth_seq_array}")

            # 对 generated 和 tgt_true 做比较
            # Confusion matrix
            T = ground_truth_seq_array + output_seq_array
            TP = (T == 2 ).sum()   # 标量 tensor
            TN = (T == 0 ).sum()
            F = ground_truth_seq_array - output_seq_array
            FN = (F == 1 ).sum()
            FP = (F == -1).sum()
            print(f"Spectrum No.{b},Cofusion matrix: TP={TP}, FP={FP}, FN={FN}, TN={TN}")
            
            # metric #1: Precision = TP/(TP+FP)
            precision = TP/(TP+FP)
            # metric #2: Recall    = TP/(TP+FN)
            recall = TP/(TP+FN)
            print(f"Precision ={precision}; Recall ={recall}")

            # save one dot in dataframe
            one_dot = pd.DataFrame(
                                    { # 🚨这里有ValueError: If using all scalar values, you must pass an index
                                    # "epoch": epoch,
                                    "recall": recall.item(),
                                    "precision": precision.item(),
                                    "TP":TP.item(), 
                                    "FP":FP.item(), 
                                    "FN":FN.item(), 
                                    "TN":TN.item(),
                                        },
                                    index=[0],)
            # add one dot into a list called "epoch_dot"
            epoch_dot.append(one_dot) # list "epoch_dot" storage all dots in one epoch
        print("this for-loop end")
        print("One batch ends.")
print("The validation loop ends")

##### Calculating metrics
# 合并所有 epoch 的结果 Combine all dots into one dataframe, called "epoch_df": epoch|recall|precision|TP|FP|FN|TN
epoch_df = pd.concat(epoch_dot, ignore_index=True) #Seven columns: epoch|recall|precision|TP|FP|FN|TN

# 计算比例 proportion of '1'
epoch_df["ratio"] = (epoch_df['TP'] + epoch_df['FN']) / ( epoch_df['TP'] + epoch_df['FP'] + epoch_df['FN'] + epoch_df['TN'])
epoch_df["seq_len"]= epoch_df['TP'] + epoch_df['FP'] + epoch_df['FN'] + epoch_df['TN']
epoch_df["peak_1_num"]=epoch_df['TP'] + epoch_df['FN']
epoch_df.columns
print(f"The dataframe infomation of 'epoch_df':\n{epoch_df.info()}")

##### 分箱; 计算count ≥ 1000 的格子

# ratio分箱
ratio_bins = pd.cut( epoch_df["ratio"],
                    bins=np.linspace(0, 1, 21),
                    include_lowest=True, )

# seq_len分箱
seq_bins = pd.cut(   epoch_df["seq_len"],
                    bins=np.linspace(0, 200, 21),
                    include_lowest=True, )

# 计算每个格子的 sample count
count_table = pd.pivot_table(
                            epoch_df,
                            values='ratio',   # 用任意非空列都可以
                            index=seq_bins,
                            columns=ratio_bins,
                            aggfunc='count',)

# 找出 count >= 1000 的 bin 组合

############📉 Epoch-level Precision-Recall scatter plot #########################
# Note:这个功能呢实现之后整合到"TrainingModel_2025_04_07.py"这个program当中
# 每个 spectrum 一个点：
# x-axis = precision
# y-axis = recall
# --------------------------------------------------

# 创建画布
plt.figure() 
# 画散点图
plt.scatter(
            epoch_df["precision"],
            epoch_df["recall"],
            s=10,
            alpha=0.05,
            )

# 固定坐标范围(必须是0~1)
plt.xlim(0, 1)
plt.ylim(0, 1)
# 坐标标签
plt.xlabel("Precision")
plt.ylabel("Recall")
# 标题
plt.title(f"Test step: Precision-Recall")
# 网格(方便观察趋势)
plt.grid(True, alpha=0.3)
# 保存高清图片
plt.savefig(
    f"{output_dir}/Test_step_Precision-Recall.png", 
    dpi=300, 
    bbox_inches='tight', )
plt.close()

# Save the result in tsv file
epoch_df.to_csv(
                f"{output_dir}/Test_step_Precision_Recall_DataFrame.tsv",
                sep="\t",
                )

######📉 heatmap: recall 
# 计算每个格子的平均 recall
recall_table = pd.pivot_table(  epoch_df,
                                values = 'recall',
                                index = seq_bins,
                                columns = ratio_bins,
                                aggfunc = 'mean',)

# seq_len 倒序显示
recall_table = recall_table.iloc[::-1]

plt.figure(figsize=(14,10))

sns.heatmap(
            recall_table,
            cmap='viridis',
            vmin=0,
            vmax=1,
            annot=False,
            cbar_kws={'label':'Mean Recall'}
        )

plt.xlabel('Ratio', fontsize=16)
plt.ylabel('Sequence Length', fontsize=16)

plt.title( f'Test_step Mean Recall Across Ratio and Sequence Length',
    fontsize=20,
    fontweight='bold'
)
plt.tight_layout()
# 保存高清图片
plt.savefig(
    f"{output_dir}/Test_step_Recall_Heatmap.png", 
    dpi=300, 
    bbox_inches='tight', )
plt.close()

###### 📉 heatmap: precision
# 计算每个格子的平均Precision
precision_table = pd.pivot_table(
    epoch_df,
    values='precision',
    index=seq_bins,
    columns=ratio_bins,
    aggfunc='mean'
)

# seq_len倒序显示
precision_table = precision_table.iloc[::-1]

plt.figure(figsize=(14,10))

sns.heatmap(
    precision_table,
    cmap='viridis',
    vmin=0,
    vmax=1,
    annot=False,
    cbar_kws={'label':'mean precison'},)

plt.xlabel( 'Ratio', fontsize=16)
plt.ylabel( 'Sequence Length', fontsize=16)
plt.title(  f'Test step: Mean Precision Across Ratio and Sequence Length',
            fontsize=20,
            fontweight='bold',)
plt.tight_layout()
# 保存高清图片
plt.savefig(
    f"{output_dir}/Test_step_Precision_Heatmap.png", 
    dpi=300, 
    bbox_inches='tight', )
# plt.show()
plt.close()

##### Epoch结算
#(II) 计算本次epoch的 valid阶段的,平均 metrics. "epoch_df" include all spectrum in validation set 
recall_average = epoch_df["recall"].mean() # recall avarage
precision_average = epoch_df["precision"].mean()# precison average


##### 打印该 epoch 的平均 loss
print(f"Test step - avarage_recall: {recall_average:.4f}, \
                    avarage_precision: {precision_average:.4f}")

print("All epoches done.")






