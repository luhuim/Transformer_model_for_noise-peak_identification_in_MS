#!/usr/bin/python3 
import pandas as pd
import torch
import torch.nn as nn
import math

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from torch.nn.utils.rnn import pad_sequence

#%% #################用于train的class#######################
# 定义一个Dataset类，用于加载和处理数据
class SpectrumDataset_new(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe  # 存储DataFrame
        print(f"Here is the input dataframe,shape:{dataframe.shape}")
    def __len__(self):
        """
        DataLoader uses this to know how many samples exist.
        """
        return len(self.dataframe)  # 返回数据集的长度

    def __getitem__(self, idx):
        """
        DataLoader calls this method to get individual samples.
        It automatically handles batching, shuffling, etc.
        """
        
        #包装spectrum peak的info,
        mz = self.dataframe.iloc[idx]['filtered_mz'].squeeze(0)  # 获取m/z数据并去掉多余的维度
        intensity_ratio = self.dataframe.iloc[idx]['filtered_intensity'].squeeze(0)# 获取intensity数据
        spectrum = torch.stack((mz, intensity_ratio), dim=1)# 创建形状为 (n_peaks, 2) 的张量,包含 m/z,intensity

        #包装 target_seq
        target_seq = self.dataframe.iloc[idx]['filtered_label'].squeeze(0)  # 获取目标序列
        
        #包装 charge
        # charge = self.dataframe.iloc[idx]['charge']# 获取charge. int64
        charge = torch.tensor(self.dataframe.iloc[idx]['charge'], dtype=torch.int32)
        #包装 pepmass
        # pepmass = self.dataframe.iloc[idx]['pepmass']# 获取pepmass float64
        pepmass = torch.tensor(self.dataframe.iloc[idx]['pepmass'], dtype=torch.float32)  # 获取pepmass信息
        # return spectrum, target_seq, charge, pepmass  # 返回spectrum和目标序列
        # 导出字典
        return {
                    'spectrum': spectrum,
                    'target_seq': target_seq,
                    'charge': charge,
                    'pepmass': pepmass
                }

# 定义collate_fn，用于填充batch中的数据
# 自定义collate_fn函数
def collate_fn(batch):
    spectra = [item['spectrum'] for item in batch]
    target_seqs = [item['target_seq'] for item in batch]
    charges = torch.stack([item['charge'] for item in batch])  # charge 是标量，直接堆叠
    pepmasses = torch.stack([item['pepmass'] for item in batch])  # pepmass 是标量，直接堆叠

    # 填充光谱谱数据 (spectrum)，使用 pad_sequence 来处理变长的张量
    spectra_padded = pad_sequence(spectra, batch_first=True, padding_value=0)
    
    # 填充目标序列，同样使用 pad_sequence，并用 -1 作为填充值
    target_seqs_padded = pad_sequence(target_seqs, batch_first=True, padding_value=-1)

    # 返回一个批次中所有的元素，继续用字典格式
    return {
        'spectrum': spectra_padded,
        'target_seq': target_seqs_padded,
        'charge': charges,
        'pepmass': pepmasses
    }

# ===== 按 peak 数量排序的 Sampler =====
class PeakLengthSortedSampler(Sampler):
    def __init__(self, dataset: SpectrumDataset_new, descending: bool = False):
        """
        dataset: 你的 SpectrumDataset_new 实例
        descending: 
            False -> 按 peak 数量从少到多
            True  -> 按 peak 数量从多到少
        """
        self.dataset = dataset

        # 预先算好每个样本的 peak 数量
        # filtered_mz 形状是 (1, n_peaks) 或 (n_peaks,)
        self.lengths = [
                        len(self.dataset.dataframe.iloc[i]['filtered_mz'].squeeze(0))
                        for i in range(len(self.dataset))
                        ]

        # 根据长度排序 index
        self.sorted_indices = sorted(
                                    range(len(self.dataset)),
                                    key=lambda idx: self.lengths[idx],
                                    reverse=descending
                                )

    def __iter__(self):
        # 按排好序的 index 依次 yield 给 DataLoader
        for idx in self.sorted_indices:
            yield idx

    def __len__(self):
        return len(self.sorted_indices)

#%% #################用于inference的class#######################################

class InferenceDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe  # 存储DataFrame
        # print("Here is __init__")

    def __len__(self):
        # print("Here is __len__")
        return len(self.dataframe)  # 返回数据集的长度

    def __getitem__(self, idx):
        
        # print("Here is __getitem__")
        
        #包装spectrum的mz
        mz = self.dataframe.iloc[idx]['filtered_mz'].squeeze(0)  # 获取m/z数据并去掉多余的维度
        #包装spectrum的intensity
        intensity_ratio = self.dataframe.iloc[idx]['filtered_intensity'].squeeze(0)# 获取intensity数据
        #把mz和intensity打包到一起
        spectrum = torch.stack((mz, intensity_ratio), dim=1)# 创建形状为 (n_peaks, 2) 的张量,包含 m/z,intensity

        #包装 charge
        # charge = self.dataframe.iloc[idx]['charge']# 获取charge. int64
        charge = torch.tensor(self.dataframe.iloc[idx]['charge'], dtype=torch.int32)
       
        #包装 pepmass
        # pepmass = self.dataframe.iloc[idx]['pepmass']# 获取pepmass float64
        pepmass = torch.tensor(self.dataframe.iloc[idx]['pepmass'], dtype=torch.float32)  # 获取pepmass信息
        
        #包装 NrScan
        NrScan = self.dataframe.iloc[idx]['ScanNr'].squeeze(0)  # spectrum ID
        # print(NrScan.dtype)
        
        # return NrScan, spectrum, charge, pepmass  # 返回 spectrum的ID和spectrum的features
        # 导出字典
        return {
                    'NrScan': NrScan,
                    'spectrum': spectrum,
                    'charge': charge,
                    'pepmass': pepmass
                }

# 定义collate_fn，用于填充batch中的数据
# 自定义collate_fn函数
def collate_fn_inference(batch):
    NrScan = [item['NrScan'] for item in batch]
    spectra = [item['spectrum'] for item in batch]
    charges = torch.stack([item['charge'] for item in batch])  # charge 是标量，直接堆叠
    pepmasses = torch.stack([item['pepmass'] for item in batch])  # pepmass 是标量，直接堆叠

    # 填充光谱谱数据 (spectrum)，使用 pad_sequence 来处理变长的张量
    spectra_padded = pad_sequence(spectra, batch_first=True, padding_value=0)
    
    # print("NrScan:",NrScan)

    # 返回一个批次中所有的元素，继续用字典格式
    return {
                "ScanNr" : NrScan,
                'spectrum': spectra_padded,
                'charge': charges,
                'pepmass': pepmasses,
            }
