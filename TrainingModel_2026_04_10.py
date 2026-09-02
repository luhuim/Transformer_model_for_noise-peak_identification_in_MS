#!/usr/bin/env python3
"""
Introduction 
    先算每个spectrum的加权算术平均loss,再算这个batch算术平均loss
    Input file: 
        2026_04_10_train_dataset.pkl
        2026_04_10_valid_dataset.pkl
"""
#%% Import packages 
print("Import packages")
import torch
from torch import nn
import torch.optim as optim

import pandas as pd
import numpy as np
import math

from dataclasses import dataclass, field
from typing import Dict, Optional

# OS module
import os
from pathlib import Path
import yaml # install command: conda install pyyaml
print("Done\n")

#%% YAML file name
config_path = "config_2026_04_10.yaml"
#%% Check GPU
from torch.cuda import amp
print("Check GPU")
#--- 设备选择 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Current device: {device}")

#Print the GPU and cuda infomation 
print("PyTorch version:", torch.__version__, "| torch.version.cuda:", torch.version.cuda)
print("Cuda available:", torch.cuda.is_available())
print("Device_count:", torch.cuda.device_count())
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
if torch.cuda.is_available():
    print("Current_device:", torch.cuda.current_device())
    print("Device_name:", torch.cuda.get_device_name(0))
    
#Command of getting GPU info
#!nvidia-smi
#This is job ID
print(f"Job ID:{os.getpid()}")
print("Done\n")

#%% YAML file, weight tensor
print("Checking the weight parameter.")
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

print(f"Weight parameter of cross entropy loss function is:{weight}")
print("Done\n")

#%% 查看checkpoint的文件夹是否存在? yes就用现成的; if no就新建一个
print("Checking the checkpoint folder name, from config file.")
checkpoint_folder_from_config = None # 初始化变量, 用于存储从 config.yaml 里读取的 folder 名称

# -------- Step 1: 读取 config（如果存在） --------
if os.path.exists(config_path): # 判断 config.yaml 文件是否存在
    try:
        with open(config_path, "r") as f:   # 以只读模式打开文件
            config = yaml.safe_load(f)      # 解析 yaml 文件为 Python 字典

        # 判断 config 是否为字典
        # 且包含 "checkpoint_folder" 这个 key
        # 且该 key 的值是字符串
        # 且字符串不为空

        if (
            isinstance(config, dict) # 判断 config 是否为字典
            and "checkpoint_folder" in config # 判断是否包含 "checkpoint_folder" 这个 key
            and isinstance(config["checkpoint_folder"], str) # 判断该 key 的值是否为字符串
            and len(config["checkpoint_folder"].strip()) > 0 # 判断字符串是否为空
        ):
            # 如果全部满足，则从 config 中读取 folder name
            checkpoint_folder_from_config = config["checkpoint_folder"] # 把字典中的"folder name"保存到变量
            print(f"Config file exists.")
            print(f'Using checkpoint folder name: "{checkpoint_folder_from_config}"')
        
        else:# config 存在但没有合法的 checkpoint_folder
            print("config.yaml found but no valid checkpoint_folder field.")
    except Exception as e: # 如果读取或解析 yaml 时发生异常(例如格式错误)
        print(f"Error reading config.yaml: {e}")
else: # 如果 config.yaml 文件不存在
    print("Config.yaml not found.")

# -------- Step 2: 决定使用哪个 folder 名称 --------
if checkpoint_folder_from_config is not None:
      # 如果成功从config文件读取到 folder name
    checkpoint_folder = checkpoint_folder_from_config
else: # 如果 config 不存在, 或 key 不存在 或格式不合法
      # 则使用自动生成逻辑
      # 获取当前.py文件的文件名(不包含扩展名)
      # 注意：__file__ 是脚本运行时才有效的变量(command: python test.py), 在交互式解释器中可能未定义(spyder,jupter notebook)
      # 如果 __file__ 未定义，则手动设置一个文件名
    if '__file__' in globals(): # 判断当前是否在脚本运行模式（例如 python train.py）
        print("One")
        current_file = __file__ # 获取当前脚本的完整路径
        print(current_file)
    else:                       # 如果在交互环境（如 Spyder / Jupyter）中运行, __file__ 变量可能不存在，因此手动指定文件名
        print("Two")
        current_file = "TrainingModel_2025_02_16.py"  #手动指定文件名 #Volley
    print(f"Current file name is: {current_file}")

    # 去掉扩展名, 例如 "train.py" -> "train"
    current_file_name = os.path.splitext(os.path.basename(current_file))[0]
                                       # os.path.basename(current_file) 从完整路径中取出文件名
                      # os.path.splitext(...) 去掉扩展名
    # 构造 checkpoint folder名称，格式为 “当前文件名 + checkpoint”
    checkpoint_folder = current_file_name + "_CheckPoint"
    
    # 打印提示
    print(f'Using auto-generated checkpoint folder: "{checkpoint_folder}"')

# -------- Step 3: 创建 checkpoint_folder 文件夹 --------
#创建该文件夹，如果已存在则不会报错
os.makedirs(checkpoint_folder, exist_ok=True)
# 打印确认信息
print(f"Checkpoint folder '{checkpoint_folder}' exist.")
print("Done\n")

#%% Open train_set.pkl file(打开二进制文件);滤掉peak数量>200的spectrum
import pickle
print("Open train_set .pkl file and load DataFrame.")
# 2026_04_10_train_dataset.pkl 
with open('2026_04_10_train_dataset.pkl', 'rb') as f: #Volley
    train_df = pickle.load(f)
print("train_set has been loaded from pkl file.")
print(f"The shape of DataFrame: {train_df.shape}") # 398,347

# 在创建Dataset之前，从train_set DataFrame里过滤掉peak数量>200的spectrum
MAX_PEAKS = 200
train_df = train_df[
                      train_df['filtered_mz'].apply(lambda x: len(x.squeeze(0)) <= MAX_PEAKS)
                      ].reset_index(drop=True)
print(f"Peak < 200, amount{len(train_df)}") # 394,574

# 从train_set DataFrame里过滤掉label全是零的spectrum
train_df = train_df[  train_df['filtered_ration_of_1s'] != 0].reset_index(drop=True)
print(f"Peak < 200; 过滤掉label全是零, amount{len(train_df)} ")# 394,491

print("Done\n")

#%% Open valid_set.pkl file(打开二进制文件);滤掉peak数量>200的spectrum
print("Open valid_set .pkl file and load DataFrame.")
# 2026_04_10_valid_dataset.pkl
with open('2026_04_10_valid_dataset.pkl', 'rb') as f: #Volley
    val_df = pickle.load(f)

print("valid_set has been loaded from pkl file.")
print(f"The shape of DataFrame: {val_df.shape}") # (74,689, 24)

# 在创建Dataset之前，从valid_set DataFrame里过滤掉peak数量>200的spectrum
MAX_PEAKS = 200
val_df = val_df[
                      val_df['filtered_mz'].apply(lambda x: len(x.squeeze(0)) <= MAX_PEAKS)
                      ].reset_index(drop=True)
print(f"Peak < 200, amount{len(val_df)}") # 73,638

# 从valid_set DataFrame里过滤掉label全是零的spectrum
val_df = val_df[  val_df['filtered_ration_of_1s'] != 0].reset_index(drop=True)
print(f"Peak < 200; 过滤掉label全是零, amount{len(val_df)}") # 73,631 

print("Done\n")

#%% Check各个数据集的大小
print(f"Train set size: {len(train_df)}")    # 394,491
print(f"Validation set size: {len(val_df)}") #  73,631

# print(train_df.columns)

#%% 创建train_dataloader的实例
# Update on 2025-10-08: checking the structure of 'pytorch.dataset' and 'pytorch.dataloader'
""" 
    Dataset的合成调用了外部的class
    
    Detaset生成的dictionary的形态为:
        -'spectrum':   spectrum (padding是0)
        -'target_seq': target_seq (padding是-1)
        -'charge': charge
        -'pepmass': pepmass
"""

print("Build train_dataloader.")
# Dataset的合成调用了外部的class
from Dataloader.Dataloader_24_06_24 import SpectrumDataset_new, collate_fn, PeakLengthSortedSampler
from torch.utils.data import DataLoader# DataLoader的合成用的是torch自带的功能

#创建train Dataset实例.Note:如果跑短的数据集,需要在这里设置DataFrame的长度train_df[0:15]
train_dataset = SpectrumDataset_new(train_df) #Volley
#创建按 peak 数量排序的 Sampler
train_sampler = PeakLengthSortedSampler(
                                        train_dataset,
                                        descending=True # False: 少→多；True: 多→少
                                        )
#创建 DataLoader, 使用自定义的collate_fn 
train_dataloader = DataLoader(
                                train_dataset,  #这里不能动
                                batch_size=32,  #这里可以改 #Volley
                                shuffle=True,   #训练阶段打乱顺序True
                                collate_fn = collate_fn,
                                # sampler=train_sampler,  # 使用 sampler 控制顺序
                                drop_last=False, # 默认不丢弃最后一个不足一个batch_size的数据
                    			num_workers=0,    	      # new
                        		# prefetch_factor=2,	  # new
                        		persistent_workers=False, # new
                        		pin_memory=False,         # new 
        			        )
print("Done\n")

#%% 创建validation_dataloader的实例
print("Build valid_dataloader.")
#创建validation Dataset实例. Note:如果跑短的数据集,需要在这里设置DataFrame的长度val_df[0:10] 
validation_dataset = SpectrumDataset_new(val_df) #Volley 

#创建DataLoader，使用自定义的collate_fn
validation_dataloader = DataLoader(
                                    validation_dataset, #这里不能动 
                                    batch_size= 16,       #这里可以改 #Volley  
                                    shuffle= False,       #valid-set要保持顺序False
                                    collate_fn=collate_fn,
                                    drop_last=False,      #默认不丢弃最后一个不足一个batch_size的数据
                            	    num_workers=0,            # new
                                    # prefetch_factor=2,      # new
                                    persistent_workers=False, # new
                                    pin_memory=False,         # new 
                		)
print("Done\n")

#%% 实例化model ============================================================================
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

#%% Define a function: save_checkpoint():
def save_checkpoint( state, file_name="checkpoint.pth"):
    print("=> Saving checkpoint")
    print("File name of checkpoint:", file_name)
    
    torch.save(state, file_name)
    return

#%% Epoch loop======================================================================================
# Sending batch into model
import torch.optim as optim
# Making plot
import matplotlib.pyplot as plt 
import seaborn as sns

# 定义 optimizer
print("Defining optimizer.")
model_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = optim.Adam(
                        model_params, 
                        lr=1e-4,
                        )  # 学习率按需要调整
print("Done\n")

# Epoch loop
print("The epoch loop")
epochs=20
log_interval= 1

#epoch iteration of model training 
epoch_train_loss_list=[] #This list saves loss of every epoch
all_epoch=[] # This list saves all DataFrame {precision, recall} of every epoch 

# epoch loop
for epoch in range(epochs):
    print(f"==== Epoch {epoch+1}/{epochs} ====")
    
    #Train mode
    encoder.train()
    decoder.train()

    # Calculating ave
    epoch_loss_sum_train = 0.0 # sum up all loss 
    total_batches_train = 0    # the number of batch
    
    #For making plot 📉
    train_loss_list=[]         # collect loss from every batch in train step, making loss curve 📉
    all_dot=[]                 # collect confusion matrix element for every spectrum in validation step 📉
    
    #train loop 
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}:")
        #==%%%%%%%%Paste%%%%%%%%%%%%%%%%%%%%%%%%%%
        # Step 1: Optimerder梯度归零
        # optimizer.zero_grad(set_to_none=True)
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
        optimizer.zero_grad() #清零梯度
        encoded_output, padding_mask =  encoder (
                                        mz_array = spectrum[:, :, 0], #把spectrum的两层拆开分别带入encoder #torch.Size([3, 43, 1])                           
                                        intensity_array = spectrum[:, :, 1],#torch.Size([3, 43, 1])
                                        pepmass = pepmass,
                                        charge = charge,
                                        mask=None,
                                        )
        
        print("encoded_output shape:",encoded_output.shape)# torch.Size([3, 44, 512])
        print("padding_mask shape:",padding_mask.shape)# torch.Size([3, 44])
        # print("encoded_output:\n",encoded_output)
        # print("padding_mask",padding_mask)
        ##################Decoder#########################################################
        #修改 start token 和 padding token
        #1) 把-1换成3(padding token)
        target_seq = torch.where(
                                target_seq == -1,
                                torch.tensor(3, device=target_seq.device),
                                target_seq
                            )
        #2) 把每一行的第一个 token 改成 2（start token）
        target_seq[:, 0] = 2
    
        # 准备喂给decoder的tensor
        tgt_input  = target_seq[:, :-1]   # [B, L_peaks],提取第1位到倒数第2位,喂给decoder
        # print(f"喂给decoder的'tgt_input',shape is:{tgt_input.shape}\n:\n {tgt_input}")
        # 准备喂给decoder的 mask
        tgt_input_mask=padding_mask[:,:-1]
        # print("tgt_input_mask shape: ",tgt_input_mask.shape)
    
        #准备正确答案用于loss function
        tgt_target = target_seq[:, 1:]    # [B, L_peaks],提取第2位到最后1位, 用来算 loss 
        #print(f"正确答案:\n {tgt_target}")
        #print(f"tgt_target is contiguous? {tgt_target.is_contiguous()}" ) #False
        #准备正确答案mask
        tgt_target_mask=padding_mask[:, 1:]
        # print("正确答案mask 'tgt_target_mask' tensor:\n",tgt_target_mask)
    
        #(II)Decoder
        print("Feed into decoder")
        logits = decoder(
                        # from encoder
                        memory = encoded_output,
                        src_key_padding_mask=padding_mask,
                        # 前半句embedding  
                        tgt_tokens=tgt_input,
                        tgt_key_padding_mask= tgt_input_mask, #######!!!!!!!!!!!!!!!!!!!!
                        )  # [B, L_peaks, 2]
        ####################################################
        print("After decoder:")
        print("(1)")
        print("logits shape  is: ",logits.shape) # [32, 106, 2]
        print("logits tensor is:\n",logits)
        # print("Is logits contiguous? ",logits.is_contiguous())
        print("(2)")
        print(f"tgt_target tensor, shape{tgt_target.shape} \n {tgt_target}") #[32, 106]

        # Initiate loss function:
        # 类别出现得越少 → weight应该越大  
        # 现在,从config.yaml文件里面读取weight的数值,weight的tensor已经在开头定义好了, 可以直接用
        # weight = torch.tensor([1.0, 9.0]).to(device, non_blocking=True) #预测错一个'1'，惩罚大约是错一个'0'的9倍

        # In CrossEntropyLoss(): 'reduction'='size_average'+'reduce'(these two are 旧接口)
        # Computing loss function: padding is 3
        loss_fn = nn.CrossEntropyLoss(
                                        weight= weight,   # weight的tensor已经在开头定义好了,可以直接用
                                                          # weight = torch.tensor([1.0, 3.8]) 预测错一个'1'，惩罚大约是错一个'0'的9倍
                                        ignore_index = 3, # padding is 3,该位置不会参与 reduction (mean / sum)
                                        reduction ='none',# The type of output: "mean" | "sum" | "none"
                                     )

        # CrossEntropyLoss 里面的三个维度
        # N = batch_size / minibatch 
        # C = 2(class 0 / class 1). "C"这一维永远是“类别维”
        # d1=序列长度

        # logits : (N, C, d1)  ,
        # target: (N, d1)

        # Calculate token-level loss（NLP）
        # logits交换维度 : (N, d1, C) = [32, 106, 2]
        logits_ce = logits.permute(0, 2, 1)   # -> (N, C, d1) = [32, 2, 106]
        print(f"After交换维度, 得到logits_ce, shape: {logits_ce.shape}")
        print("Done\n")

        print("喂给 Loss function, 得到Loss_matrix")
        # 喂给loss function : loss = criterion(logits_ce, target)
        loss_matrix = loss_fn( logits_ce, tgt_target)
        print(f"Loss matrix:\nshape:{loss_matrix.shape}\n{loss_matrix} ") #loss的shape: torch.Size([])
        print("Done\n")

        print("先算每个spectrum的加权算术平均loss,再算这个batch算术平均loss")
        def compute_weighted_mean( 
                                    loss_matrix, 
                                    target, 
                                    weight, 
                                    ignore_index
                                    ):
            """
            先算每个spectrum的加权算术平均loss,再算这个batch算术平均
            :param loss_matrix: reduction='none'时输出的损失矩阵, 形状和target一致
            :param target: 标签矩阵,包含0/1/ignore_index
            :param weight: 类别权重，形状[num_classes]
            :param ignore_index: 要忽略的标签值
            :return: 加权平均损失（标量）
            """
            # 步骤1：创建掩码，标记出【有效位置】（排除ignore_index的位置）
            print("Step 1: 创建掩码 True= spectrum peaks,False=padding")
            valid_mask = (target != ignore_index)  # True=参与计算，False=忽略
            if not valid_mask.any():  # 防止全是填充，除零报错
                return torch.tensor(0.0, device=loss_matrix.device, dtype=loss_matrix.dtype)
            print(f"Valid_mask\nshape:{valid_mask.shape}\ntensor:{valid_mask}")
            print("Done\n")

            # 步骤2：获取每个有效位置对应的【类别权重】
            print("Step 2: 获取每个有效位置对应的类别权重")
            per_pos_weight = torch.zeros_like(target, dtype=torch.float32)
            per_pos_weight[valid_mask] = weight[target[valid_mask]]  # 核心：按类别分配权重
            print(f"类别权重 per_pos_weight\nshape:{per_pos_weight.shape}\ntensor{per_pos_weight}")
            print("Done\n")


            # 步骤3: 计算每一个spectrum的 weighted_loss_sum,和weight_sum
            print("Step 3: 计算每一个spectrum的 weighted_loss_sum,和weight_sum")
            sum_weighted_loss = torch.sum(loss_matrix,   dim=1)
            sum_total_weight =  torch.sum(per_pos_weight,dim=1)  # 关键：分母是权重和，不是样本数
            print(f"分子 array(loss):\n{sum_weighted_loss}\n分母 array(weight):\n{sum_total_weight}")
            print("Done\n")

            # 步骤5：计算加权算术平均 = 加权损失总和 / 有效位置的权重总和
            print("Step 5: 计算加权算术平均array = 加权损失总和array / 有效位置的权重总和array") 
            weighted_mean = sum_weighted_loss / sum_total_weight
            print(f"加权算术平均array\nshape:{weighted_mean.shape}\nvalue:{weighted_mean}")
            print("Done\n")
        
            # 计算这个batch里面算术平均loss
            print("Step 6: 计算这个batch里面算术平均loss")
            average = torch.mean(weighted_mean.float())  # 转成浮点型再计算，避免整数截断
            print(f"Arithmetic mean loss of this batch {average}")
            print("Done\n")

            return average

        # Run this function
        manual_weighted_mean = compute_weighted_mean(
                            loss_matrix=loss_matrix,
                            target=tgt_target,
                            weight=weight,
                            ignore_index=3,
                        )
        print("Arithmetic mean loss of this batch:", manual_weighted_mean)
        print("Done")

        # Assign Arithmetic mean loss as final "loss"
        print("Assign Arithmetic mean loss as final 'loss'.")
        loss= manual_weighted_mean
        print("Done")

        # Backward propragation 
        loss.backward() 
        print("Backward: Done")
        optimizer.step() 
        print("Optimizer: Done")
        
        # record the loss
        loss_value =loss.item()
        epoch_loss_sum_train += loss_value # sum up the loss value
        total_batches_train += 1           # the number of batch

        # every iter record Loss
        if i % log_interval == 0:
            train_loss_list.append(loss_value) # Add loss value into list "train_loss_list", making loss curve plot.
        print("Done\n")        

        # Print out some messages.
        print(f"Loss of batch {i}: {loss}")
        print("One batch done.")
    print("Training loop finished.")

    ###########📉损失函数随迭代次数变化的曲线图(Loss Curve)"""
    # train_loss_list, 你的 loss list
    # 创建画布
    plt.figure(figsize=(10,6))
    # making plot
    plt.plot(train_loss_list, label="Training set", linewidth=1)
    #plt.plot(valid_losses, label="Verification set", linewidth=1)
    
    # 坐标标签
    plt.xlabel("Iterations, batch_size=32", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    # 标题
    plt.title(f"Loss Over Iterations: Epoch {epoch}", fontsize=14)
    
    plt.legend()
    # 网格(方便观察趋势)
    plt.grid(True, alpha=0.3)
    
    # 固定坐标范围(必须是0~1)
    plt.ylim(0, 1) # 固定y轴范围
    # 保存图片
    plt.savefig(
        f"{checkpoint_folder}/loss_curve_Epoch{epoch}.png",
        dpi=300, 
        bbox_inches='tight',)
    #####################################################################################

    print("保存Checkpoint的.pth文件")
    """保存Checkpoint的步骤,共三步.(start) """ #保存本次epoch的checkpoint到某个文件当中
    #(1/3)做字典
    checkpoint_dic={
        'epoch':epoch, #当前的epoch数
        # 'model_state_dict': model.state_dict(),
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'loss': epoch_loss_sum_train, #当前积累的loss总和
        }
    #(2/3)做一个pth文件名
    checkpoint_file_name=f"checkpoint_epoch{epoch}.pth"#文件名
    #(3/3)调用保存checkpoint文件的这个function: 把'字典'放入'pth文件'
    save_checkpoint(
                    checkpoint_dic, # checkpoint这个字典 
                    file_name=checkpoint_folder+'/'+ checkpoint_file_name#文件名
                   #file_name是一个完整路径:'文件夹name' + '/' + '文件name'
                    )
    """保存Checkpoint (End)"""

    ########################Valid_step#######################################################################
    print("Move to validation step")
    # Validation mode, 此处'.eval()'的作用是:关闭 dropout、batchnorm 的训练行为
    encoder.eval() 
    decoder.eval()
    print("validation mode")
    # 清零 every epoch
    epoch_dot=[]        # 本次epoch的,每个spectrum的,混淆矩阵 and metrics. one dataframe, one spectrum

    with torch.no_grad(): #不计算梯度
        for i, batch in enumerate(validation_dataloader):# validation_dataloader
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
                                        "epoch": epoch,
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

    #######📉 Epoch-level Precision-Recall scatter plot #########################
    # Note:这个功能呢实现之后整合到"TrainingModel_2025_04_07.py"这个program当中
    # 每个 spectrum 一个点：
    # x-axis = precision
    # y-axis = recall
    # --------------------------------------------------
    # 合并所有 epoch 的结果 Combine all dots into one dataframe, called "epoch_df": epoch|recall|precision|TP|FP|FN|TN
    epoch_df = pd.concat(epoch_dot, ignore_index=True) #Seven columns: epoch|recall|precision|TP|FP|FN|TN
    # 计算比例 proportion of '1'
    epoch_df['ratio'] = (epoch_df['TP'] + epoch_df['FN']) / ( epoch_df['TP'] + epoch_df['FP'] + epoch_df['FN'] + epoch_df['TN'])
    epoch_df["seq_len"]= epoch_df['TP'] + epoch_df['FP'] + epoch_df['FN'] + epoch_df['TN']
    epoch_df["peak_1_num"]=epoch_df['TP'] + epoch_df['FN']
    epoch_df.columns
    
    print(f"The dataframe infomation of 'epoch_df':\n{epoch_df.info()}")
    # 创建画布 📉
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
    plt.title(f"Epoch:{epoch} Precision-Recall")
    # 网格(方便观察趋势)
    plt.grid(True, alpha=0.3)
    # 保存高清图片
    plt.savefig(
        f"{checkpoint_folder}/Epoch:{epoch}_Precision-Recall.png", 
        dpi=300, 
        bbox_inches='tight', )

    #Save the result in tsv file
    epoch_df.to_csv(
                    f"{checkpoint_folder}/Epoch:{epoch}_Precision_Recall_DataFrame.tsv",
                    sep="\t",
                    )
    
    ######📉 heatmap: recall 
    # ratio分箱
    ratio_bins = pd.cut( epoch_df['ratio'],
                        bins=np.linspace(0, 1, 21),
                        include_lowest=True, )

    # seq_len分箱
    seq_bins = pd.cut(   epoch_df['seq_len'],
                        bins=np.linspace(0, 200, 21),
                        include_lowest=True, )

    # 计算每个格子的平均 Recall
    recall_table = pd.pivot_table(  epoch_df,
                                    values='recall',
                                    index=seq_bins,
                                    columns=ratio_bins,
                                    aggfunc='mean',)

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

    plt.title( f'Epoch:{epoch} Mean Recall Across Ratio and Sequence Length',
        fontsize=20,
        fontweight='bold'
    )
    plt.tight_layout()
    # 保存高清图片
    plt.savefig(
        f"{checkpoint_folder}/Epoch:{epoch}_Recall_Heatmap.png", 
        dpi=300, 
        bbox_inches='tight', )
    
##### 📉 heatmap: precision
    # ratio分箱
    ratio_bins = pd.cut( epoch_df['ratio'],
                        bins=np.linspace(0, 1, 21),
                        include_lowest=True, )

    # seq_len分箱
    seq_bins = pd.cut(   epoch_df['seq_len'],
                        bins=np.linspace(0, 200, 21),
                        include_lowest=True, )

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
    plt.title(  f'Epoch:{epoch} Mean Precision Across Ratio and Sequence Length',
                fontsize=20,
                fontweight='bold',)
    plt.tight_layout()
    # 保存高清图片
    plt.savefig(
        f"{checkpoint_folder}/Epoch:{epoch}_Precision_Heatmap.png", 
        dpi=300, 
        bbox_inches='tight', )
    # plt.show()

    """Epoch结算"""
    # 计算本次epoch的,train阶段的,平均loss: 训练的总loss / 训练的batch数
    avg_train_loss = epoch_loss_sum_train / total_batches_train
    # save in dataframe
    one_epoch_train_df = pd.DataFrame(
                            {
                            "epoch": epoch,
                            "average_loss": avg_train_loss,
                            },
                            index=[0],)
    # add "one_epoch_train_df" into list
    epoch_train_loss_list.append(one_epoch_train_df) # add this DataFrame into a list "epoch_train_loss_list"

    # 计算本次epoch的,valid阶段的,平均metrics. "epoch_df" include all spectrum in validation set 
    recall_average = epoch_df["recall"].mean() # recall avarage
    precision_average = epoch_df["precision"].mean()# precison average
    
    # save in dataframe
    after_one_epoch = pd.DataFrame(
                            {
                            "epoch": epoch,
                            "recall": recall_average.item(),
                            "precision": precision_average.item(),
                            },
                            index=[0],)
    # all_df.append(df) all_dot=[]
    all_epoch.append(after_one_epoch)# list "all_dot" storage all dots 

    # 打印该 epoch 的平均 loss
    print(f"Epoch {epoch} - Training Loss: {avg_train_loss:.4f}; Validation avarage_recall: {recall_average:.4f}, avarage_precision: {precision_average:.4f}")
print("All epoches done.")

#%% 📉 Epoch-level Precision-Recall scatter plot
# 每个 epoch 一个点：
# x-axis = precision
# y-axis = recall
# --------------------------------------------------

# 合并所有 epoch 的结果 Combine all dots into one dataframe, epoch | recall | precision
all_epoch_df = pd.concat(all_epoch, ignore_index=True) #Three columns: epoch, recall, precision

# 创建画布
plt.figure() 
# 画散点图
plt.scatter(
            all_epoch_df["precision"],
            all_epoch_df["recall"],
            s=20)
# 在每个点旁标注 epoch 编号
for i, row in all_epoch_df.iterrows():
    plt.text(row["precision"],
             row["recall"],
             str(int(row["epoch"])),
             fontsize=6)
# 固定坐标范围(必须是0~1)
plt.xlim(0, 1)
plt.ylim(0, 1)
# 坐标标签
plt.xlabel("Mean Precision")
plt.ylabel("Mean Recall")
# 标题
plt.title("Epoch-level Precision-Recall")
# 网格(方便观察趋势)
plt.grid(True, alpha=0.3)
# 保存高清图片
plt.savefig(
    f"{checkpoint_folder}/Epoch_level_Precision-Recall.png", 
    dpi=300, 
    bbox_inches='tight', )
# 显示图片
plt.show()

#%% Save some statistics in tsv file

# Train step: save average loss 
# 合并所有 train epoch 的结果 Combine all dots into one dataframe, epoch | avarage_loss
output_df_train = pd.concat(epoch_train_loss_list, ignore_index=True) #Two columns: epoch, avarage_loss
# Save in tsv file
output_df_train.to_csv(
                        f"{checkpoint_folder}/Train_set_result_over_epochs.tsv",
                        sep="\t", 
                        )
print(f"Saved in file: {checkpoint_folder}/Train_set_result_over_epochs.tsv")

# Valid Step: saving average precision and recall value in tsv file
all_epoch_df.to_csv(
                    f"{checkpoint_folder}/Valid_set_result_over_epochs.tsv",
                    sep="\t",
                    )
print(f"Saved in file: {checkpoint_folder}/Valid_set_result_over_epochs.tsv")



