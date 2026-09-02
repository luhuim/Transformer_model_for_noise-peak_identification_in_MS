#!/usr/bin/env python3
"""
Title: class of decoder
Date:  2026 feb 20
Author: Huimin Lu

Introduction:
    This is the class of decoder.
    This generator is simple and straight forward: logits --> softmax() --> argmax() --> next_token

Path: MasterThesis/RunOnServer/TransformerBlock/Decoder_25_12_08.py

Usage:
    from TransformerBlock.Decoder_25_12_08 import TransformerDecoderCaseB, PositionalEncoding
"""
#%% 对0和1组成的target seq进行positional encoding PositionalEncoding()
import torch
import torch.nn as nn
import math
# Positional Encoding class as defined earlier
class PositionalEncoding(nn.Module):
    def __init__(self, 
                 d_model, 
                 dropout=0.1, 
                 max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # Register buffer so that it is part of the model but not learnable
        self.register_buffer('pe', pe)
    def forward(self, x):
        # Add positional encoding to the input tensor
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
#%% 
class TransformerDecoderCaseB(nn.Module):
    """
    用于训练的 Transformer decoder（情况 B）：
        - vocab = {0, 1}，用 CrossEntropyLoss 训练
                            0 → token=0
                            1 → token=1
                            2 → BOS
        - 输入:  tgt_tokens (teacher forcing)
        - 输出:  logits, 形状 [B, L_peaks, 2]
    
    Binary classification 的两种常见做法
        -✅2-class softmax: logits.shape = (B,2) + softmax + argmax
        -  1-logit sigmoid: logits.shape = (B,1) + sigmoid + p>0.5
    """
    def __init__(
            self,
            d_model: int,
            nhead: int,
            num_layers: int,
            dim_feedforward: int,
            dropout: float = 0.1,
            bos_token: int = 2,     # 起始 token，沿用你之前的做法
                ):
        super().__init__()

        self.d_model = d_model
        self.bos_token = bos_token
        # self.pad_token = pad_token
        
        # 1) TransformerDecoder 层
        decoder_layer = nn.TransformerDecoderLayer(
                                                    d_model=d_model,
                                                    nhead=nhead,
                                                    dim_feedforward=dim_feedforward,
                                                    dropout=dropout,
                                                    batch_first=True, # 这样所有输入输出都是 [B, L, D]
                                                )
        #
        self.transformer_decoder = nn.TransformerDecoder(
                                                        decoder_layer,
                                                        num_layers=num_layers
                                                        )

        # 2) token embedding：vocab_size=3 : token 0 ; token 1; 2 BOS
        self.embedding = nn.Embedding(
                                        num_embeddings=4,    # 0 和 1 and 2, 3 
                                        embedding_dim=d_model,
                                        padding_idx=None
                                        )
        
        # 3) 位置编码
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        # 4) 输出层：对每个位置输出 2 维 logits (0 / 1)
        self.output_layer = nn.Linear(d_model, 2)

    def _generate_tgt_mask(self, tgt_len: int): #生成causal_mask,正方形 
        """
            - 生成标准的causal mask(下三角),保证位置i只看见<=i的token,每个位置i通过mask的第i行,从tgt_emb获取的可见范围,
            - 因为从外面喂进decoder的是整个长度的tensor, 但是预测不同位置的token需要带入不同长度的embedding tensor
            - 所以,causal mask是用来切割每个position的embedding tensor
        """
        return nn.Transformer.generate_square_subsequent_mask( tgt_len )

    def forward(
        self,
        tgt_tokens: torch.Tensor,                       # (batch, tgt_len), teacher forcing 输入
        memory: torch.Tensor,                           # (batch, seq_len, d_model)，encoder生成的memory tensor
        # 两种padding
        src_key_padding_mask: torch.Tensor = None,      # (batch, seq_len), encoder生成的memory的padding,True为padding 
        tgt_key_padding_mask: torch.Tensor = None,      # (batch, tgt_len), tgt_tokens的padding !!!!!!How it works?????
                    ):
        """
        训练模式：
        - 输入: tgt_tokens 是teacher forcing 输入序列, 
                           通常是 ground truth 序列去掉最后一位，如 <BOS> y0 y1 ... y_{L-2}
                           
        - 输出: 每个位置对 0/1 的 logits: (batch, tgt_len, 2)
        """
        print("Here is decoder.forward.")
        B, L_tgt = tgt_tokens.size()

        # if tgt_key_padding_mask is None:
        #     tgt_key_padding_mask = (tgt_tokens == self.pad_token)
            
        # 1) embedding + 缩放scaling: 
        #    embedding: 提供 token 的“语义”信息（0/1/BOS）
        #    缩放scaling:让embedding的数值尺度与positional encoding的值尺度一致
        #                             防止两者叠加时 PE 的影响被掩盖
        tgt_emb = self.embedding(tgt_tokens) * math.sqrt(self.d_model)
        # 2) 位置编码
        tgt_emb = self.positional_encoding(tgt_emb)

        # 3) Causal mask（self-attention 用）(保证自回归预测) 第i个位置只能看到自己和之前的位置
        tgt_mask = self._generate_tgt_mask(L_tgt) # [L, L]


        # 4) Decoder: cross-attention + self-attention
        out = self.transformer_decoder(
              # 第一部分: masked self-attention, 第i个token根据前面所有token的历史信息来预测下一个token
              tgt=tgt_emb,               # 位置编码之后的tensor, [B, L, D]
              tgt_mask=tgt_mask,         # 每个位置只能看见当前和前面的位置
              #第二部分: cross-attention, 让decoder的每个position都能参考encoder生成的一整个memory (tensor)
              # 所以无论encoder做出来的memory tensor的长度有多夸张,都不影响decoder的计算.
              # 因为cross attention的计算是: 这个spectrum的每一个position的hidden tensor v.s. 整个memory tensor
              #                             换句话说就是"1对多",求总和,所以多长也无所谓.
                                      #   query   来自 decoder 的 tgt_emb  
                                      # key/value 来自 encoder 的 memory
                                      # memory_key_padding_mask 告诉decoder哪些encoder位置是 padding
              memory=memory,             # [B, L_src, D]
              memory_key_padding_mask=src_key_padding_mask,  # [B, L_src] 或 None
              #  这里暂不使用 tgt_key_padding_mask，如有需要可以补
                                      ) # (batch, tgt_len, d_model)

        logits = self.output_layer(out)  # (batch, tgt_len, 2)
        return logits
    
    @torch.no_grad()
    def generate(
                self, 
                memory,              #来自encoder的输出              (batch_size, enc_seq_len, d_model)
                src_key_padding_mask,#来自encoder的输出的padding mask (batch_size, enc_seq_len)
                greedy=True,
                # max_seq_len,       #It will be set inside of the generator() 
                ):
        """
        What tasks are in the generator ?
            [x] Step 0: 获得device
            [x] Step 1: obtain the batch_size, and length
            [x] Step 2: making the first sequence tensor (batch_size, 1)
            [x] Step 3: time step for-loop, extending one token each time
                    (x) for t in range (0, length-1 ): 
                    (x) print("time step: ",t)
                    (x) the length of current tensor
                    (x) embedding the seqeunce tensor (it might include 0, 1, and 2) you don't have to do with padding 
                    (x) position encoding 
                    (x) making square mask
                    (x) feed this embedding into decoder, the result is a tensor (batch_size,length, 512)
                    (x) collect the final token of decoder  (dim=512)
                    (x) feed tensor into output layer, the result is logits(dim=2).(the value in padding exist but they are useless )
                    (x) predict the value for the logits
                    (x) adding value into sequence
            [x] Step 4: return the seq_tensor 
        """
        print("Here is decoder.generate.")
        #Step 0:获得device
        device = memory.device

        #Step 1:获得 batch_size和 batch_length长度
        print("Decoder step 1: 获得 batch_size和 batch_length长度")
        batch_size   = memory.size(0)
        batch_length = memory.size(1) # the length of this batch 
        #Step 2:Making the start_token tensor (batch_size, 1)
        sequence = torch.full( 
                                (batch_size, 1),            # 创建一个大小为(batch_size, 1)的tensor
                                fill_value=self.bos_token,  # 把张量里所有位置都填成bos, 例如: 如果 bos_token=2,那这个张量里都是2
                                dtype=torch.long,           # Transformer 的 token 索引必须是 long 类型（整数类型）
                                device=device,              # 确保张量创建在CUDA或CPU上,与模型一致
                                )
        #Step 3: time step for-loop, extending one token each time, 只要尚有未完成的样本且未到 batch_length，就继续生成
        print("Step 3: time step for-loop")
        for t in range (batch_length -1): # 此处的"-1"是因为少算1位token; 
            # print("time_step:",t)
            # getting the length of sequence tensor
            seq_len = sequence.size(1) 
            # embedding the seqeunce tensor
            seq_emb = self.embedding(sequence) * math.sqrt(self.d_model)
            # going through positional encoding
            seq_emb = self.positional_encoding(seq_emb)
            # making square mask
            seq_mask = self._generate_tgt_mask(seq_len)
            # feed into decoder
            out = self.transformer_decoder( #
                                            tgt = seq_emb,       # seqeunce 
                                            tgt_mask = seq_mask, # that square matrix
                                            # tgt_key_padding_mask=(tgt_tokens == self.pad_token), #I don't know
                                            #
                                            memory=memory,       #from encoder
                                            memory_key_padding_mask=src_key_padding_mask, #from encoder 
                                            )# (batch_size, tgt_len, d_model)
            
            # extract the final token, which represents newly generated token in this time-step   
            last_h = out[:, -1, :]                              # (batch_size, d_model)
            # feed the tensor of newly generated token into output layer, getting the logits
            logits = self.output_layer(last_h)                  # (batch_size, 2)

            # prediction the logits, and obtain the "next_token"
            if greedy:
                prob = logits.softmax(dim=-1)  # (B, 2)
                next_token = prob.argmax(dim=-1, keepdim=True)       # (batch_size,1), 值为0或1
                print(f"New token: {next_token.shape};\n{next_token}")

            # adding value into sequence
            sequence = torch.cat([sequence, next_token], dim=1)
        print("time step for-loop end.")
        return sequence[:, 1:]