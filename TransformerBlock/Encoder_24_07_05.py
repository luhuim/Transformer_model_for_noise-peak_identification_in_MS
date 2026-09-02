# /usr/bindaoriu/python3
#%% 
#导入package
import torch
from torch import nn
from typing import Union
from typing import Optional
from collections.abc import Callable

#导入写好的block
from TransformerBlock.PositionalEncoding_24_07_02 import PeakEncoder,ModelMixin #导入位置编码
from TransformerBlock.PositionalEncoding_24_07_02 import TransformerMixin,MassEncoder #导入encoder

class SpectrumTransformerEncoder(
                            torch.nn.Module, 
                            ModelMixin, 
                            TransformerMixin
                                    ):
    """
    A Transformer encoder for input mass spectra.
        - Use this PyTorch module to embed mass spectra. 
        - By default, nothing other than the m/z and intensity arrays for each mass spectrum are considered.    
        - However, arbitrary information can be integrated into the spectrum representation by 
            subclassing this class and overwriting the `global_token_hook()` method.

    Parameters
    ----------
    d_model : int, optional (质谱peaks的潜在维度)
        The latent dimensionality to represent peaks in the mass spectrum.
    
    nhead : int, optional (每一层有几个头，)
        The number of attention heads in each layer. 
        ``d_model``(peak的维度) must be divisible by ``nhead``(头的个数).
    
    dim_feedforward : int, optional  (全链接层的维度)
        The dimensionality of the fully connected layers in the Transformer layers of the model.
    
    n_layers : int, optional  (Transformer的层数)
        The number of Transformer layers.
    
    dropout : float, optional
        The dropout probability for all layers.
    
    peak_encoder : PeakEncoder or bool, optional
        The function to encode the (m/z, intensity) tuples of each mass spectrum. (encode用哪个公式)
            - `True` uses the default sinusoidal encoding (true 就用默认的simusoidal encoding)
            - `False`instead performs a 1 to `d_model` learned linear projection.

    Attributes
    ----------
    d_model : int
    nhead : int
    dim_feedforward : int
    n_layers : int
    dropout : float
    peak_encoder : torch.nn.Module or Callable
        The function to encode the (m/z, intensity) tuples of each mass spectrum.
    transformer_encoder : torch.nn.TransformerEncoder
        The Transformer encoder layers.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        n_layers: int = 2,
        dropout: float = 0.0,
        max_charge: int =10,
        peak_encoder: Union[PeakEncoder, Callable, bool] = True,#spectrum 的位置编码
        mass_encoder: Union[MassEncoder, Callable, bool] = True #pepmass的位置编码
                ) -> None:
        
        """Initialize a SpectrumEncoder."""
        super().__init__()
        
        self._d_model = d_model
        self._nhead = nhead
        self._dim_feedforward = dim_feedforward
        self._n_layers = n_layers
        self._dropout = dropout
        self.max_charge = max_charge

        
        """判断“peak_encoder”是否存在,以及初始化"""
        if callable(peak_encoder):
            self.peak_encoder = peak_encoder
        elif peak_encoder:#走的是这条线
            self.peak_encoder = PeakEncoder(d_model)
        print("Initializing 'peak_encoder', Done")
        
        """判断“pepmass_encoder”是否存在,以及初始化"""
        if callable(mass_encoder):
            self.mass_encoder = mass_encoder
        elif mass_encoder:#走的时这条线
            self.mass_encoder = MassEncoder(dim_model=d_model)
        print("Initializing 'mass_encoder', Done")
        
        """初始化charge的位置编码"""
        self.charge_encoder=nn.Embedding(max_charge, d_model)
        print("Initializing 'charge_encoder', Done")
        
        """定义transformers encoder里面的一层"""
        # The Transformer encoder layers:
        layer = torch.nn.TransformerEncoderLayer( #用的是pytorch的transformers
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            batch_first=True,
            dropout=self.dropout,
        )
        print("Initialize one encoder 'layer', Done")
        
        """定义一个encoder, 里面有若干层layer"""
        self.transformer_encoder = torch.nn.TransformerEncoder(
                                                                layer,
                                                                num_layers=self.n_layers,
                                                            )
        print("Initializing one 'transformer_encoder', Done")
        
    def forward(
        self,
        mz_array: torch.Tensor, # m/z的tensor
        intensity_array: torch.Tensor, # intensity的tensor
        pepmass: torch.Tensor,  # 添加 pepmass的tensor 作为输入参数
        charge:  torch.Tensor,  # 添加 charge tensor作为输入参数
        *args: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        # mask=None,  # 不需要额外的遮挡 mask
        **kwargs: dict,
                        ) -> tuple[torch.Tensor, torch.Tensor]:

        """ Embed a batch of mass spectra. Embed一批质谱
        
        Parameters
        ----------
        
        mz_array (对m/z的要求): torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
            
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
        
            The zero-padded intensity dimension for a batch of mass spctra.
        
        *args : torch.Tensor
            Additional data. These may be used by overwriting the
            `global_token_hook()` method in a subclass.
        mask : torch.Tensor
            Passed to `torch.nn.TransformerEncoder.forward()`. The mask for the sequence.
        
        **kwargs : dict
            Additional data fields. These may be used by overwriting
            the `global_token_hook()` method in a subclass.

        Returns
        -------
        latent : torch.Tensor of shape (n_spectra, n_peaks + 1, d_model)
            The latent representations for the spectrum and each of its
            peaks.
            
        mem_mask : torch.Tensor
            The memory mask specifying which elements were padding in X.
        """
        #Step1,把mz和intensity放到一个"栈"里面
        spectra = torch.stack([mz_array, intensity_array], dim=2)

        # Step2, Create the 2 种 padding mask: 质谱的mask 和 全局token的mask
        # (1/2) "src_key_padding_mask"
        src_key_padding_mask = spectra.sum(dim=2) == 0
        # print(f"dtype is:{src_key_padding_mask.dtype}; shape is {src_key_padding_mask.shape}; device is {src_key_padding_mask.device}")
        # 布尔值矩阵 src_key_padding_mask 主要用于指示哪些位置是填充的,即全为 0 的峰值,后面可以忽略
        
        # (2/2) "global_token_mask"
        # print("global_token_mask")
        #Old version
        global_token_mask = torch.tensor([[False]] * spectra.shape[0]).type_as(
                                        src_key_padding_mask)
        # print(f"Old versiion, dtype is:{global_token_mask.dtype}; shape is {global_token_mask.shape}; device is {global_token_mask.device}")

        #New version
        device = spectra.device
        global_token_mask = torch.zeros(
                                            (spectra.size(0), 1),
                                            dtype=src_key_padding_mask.dtype,
                                            device=device
                                        )
        # print(f"New versiion, dtype is:{global_token_mask.dtype}; shape is {global_token_mask.shape}; device is {global_token_mask.device}")
        #

        # print("z")
        # print("global_token_mask shape:",global_token_mask.shape)
        # print("global_token_mask:\n",global_token_mask)
        #"global_token_mask"是一个形状为(n_spectra, 1)的张量,值全为 False
        #用于在 Transformer 中为每个质谱添加一个 "全局 token",这个 token 通常表示质谱整体的信息，而不是某个特定的峰值。它不会被填充屏蔽，因此被标记为 False.
        
        #Step2.5 把两种mask结合起来, 全局token的mask在前,spectrum的mask在后
        src_key_padding_mask = torch.cat(
            [global_token_mask, src_key_padding_mask], dim=1
        )
        # print(f"src_key_padding_mask; dtype is:{src_key_padding_mask.dtype}; shape is {src_key_padding_mask.shape}; device is {src_key_padding_mask.device}")
        # print("Step2.5 两种mask结合之后,src_key_padding_mask:\n",src_key_padding_mask.shape)
        
        #Step3,把spectra(那个"栈")的tensor带入"positional encoding"
        # print("Step3,把spectra(那个栈)的tensor带入'positional encoding'")
        peaks = self.peak_encoder(spectra) #把m/z和intensity一起放进入positional encoding
        # print("Positional encoding")
        # print(f"dtype is:{peaks.dtype}; shape is {peaks.shape}; device is {peaks.device}")
        #结果的形状通常为(n_spectra, n_peaks, 512)
        # print("spectra位置编码之后的tensor的shape:",peaks.shape)
        
        
        
        # Step 4, Add the precursor information: 添加额外信息 全局token
        #默认情况下,global_token_hook() 返回一个形状为 (batch_size, 512) 的张量
        # print("Step 4, Add the precursor information: 添加额外信息 全局token,embedding")
        # 旧的
        # latent_spectra = self.global_token_hook(
        #     pepmass=pepmass,  # 传入 pepmass
        #     charge=charge,    #传入 charge
        #     *args,
        #     **kwargs,
        # )
        # 新的
        latent_spectra = self.global_token_hook(
                                                mz_array=mz_array,
                                                intensity_array=intensity_array,
                                                pepmass=pepmass,
                                                charge=charge,
                                                *args,
                                                **kwargs,
                                                    )
        # print("全局token的embedding的shape:",latent_spectra.shape)
        # print(f"latent_spectra; dtype is:{latent_spectra.dtype}; shape is {latent_spectra.shape}; device is {latent_spectra.device}")
        
        # print("Step 5, 把全局token的embedding和spectrum的位置编码tensor拼接在一起,组生成新的tensor'peaks'")
        # Step 5, 把spectrum的位置编码tensor和额外信息tensor叠加在一起，生成新的tensor
        peaks = torch.cat([latent_spectra[:, None, :], #全局hook的tensor(n_spectra, 1, 512)
                           peaks], # positional encoding之后的tensor    (n_spectra, n_peaks, 512)
                          dim=1)#cat拼接
        # print(f"peaks; dtype is:{peaks.dtype}; shape is {peaks.shape}; device is {peaks.device}")
        
        #Step 6, 把这个新的tensor,叫peaks,放入encoder
        # print("Step 6, 把这个新的tensor'peaks',放入encoder")
        #把positional embedding的那一堆东西放到transformers的encoder当中
        out = self.transformer_encoder(
            peaks,#结合起来的embedding
            # mask=src_key_padding_mask,#试一下吧 不行就删掉这一行
            src_key_padding_mask=src_key_padding_mask, #两种mask结合之后的mask
        )
        # print(f"out; dtype is:{out.dtype}; shape is {out.shape}; device is {out.device}")
        return out, src_key_padding_mask
        
    def global_token_hook(
        self,
        mz_array: torch.Tensor,
        intensity_array: torch.Tensor,
        #全局token成分 pepmass和charge
        pepmass: torch.Tensor,# pepmass
        charge:  torch.Tensor,# charge
        # max_charge: int =10,#init里面已经提过了
        # d_model: int = 512, #init里面已经提过了
        *args: torch.Tensor,
        **kwargs: dict,
                    ) -> torch.Tensor:
        
        """
        Define how additional information in the batch may be used.

        Overwrite this method to define custom functionality dependent on
        information in the batch. Examples would be to incorporate any
        combination of the mass, charge, retention time, or
        ion mobility of a precursor ion.

        The representation returned by this method is preprended to the
        peak representations that are fed into the Transformer encoder and
        ultimately contribute to the spectrum representation that is the
        first element of the sequence in the model output.

        By default, this method returns a tensor of zeros.

        Parameters
        ----------
        mz_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded m/z dimension for a batch of mass spectra.
        intensity_array : torch.Tensor of shape (n_spectra, n_peaks)
            The zero-padded intensity dimension for a batch of mass spctra.
        *args : torch.Tensor
            Additional data passed with the batch.
        **kwargs : dict
            Additional data passed with the batch.

        Returns
        -------
        torch.Tensor of shape (batch_size, d_model)
            The precursor representations.

        """
        
        #添加 pepmass和charge的encoding
        print("charge_embedding")
        charge_embedding = self.charge_encoder(charge)
        print(f"dtype is:{charge_embedding.dtype}; shape is {charge_embedding.shape}; device is {charge_embedding.device}")
        # print("charge_embedding shape:",charge_embedding.shape)#torch.Size([3, 512])
        
        pepmass_embedding = self.mass_encoder(pepmass)
        # print("pepmass_embedding shape(1):",pepmass_embedding.shape)#torch.Size([3, 512])
        
        # 将质量编码和电荷编码相加，得到前体表示 torch.Size([3, 512])+torch.Size([3, 512])
        precursors = pepmass_embedding + charge_embedding# (batch_size, dim_model)
        # print("precursors shape:",precursors.shape) # torch.Size([3, 512])
        
        return precursors
#%% encoder for mass 
import numpy as np
class MassEncoder(torch.nn.Module):
    """Encode mass values using sine and cosine waves.

    Parameters
    ----------
    dim_model : int
        The number of features to output.
    min_wavelength : float
        The minimum wavelength to use.
    max_wavelength : float
        The maximum wavelength to use.
    """

    def __init__(self, dim_model, min_wavelength=0.001, max_wavelength=10000):
        """Initialize the MassEncoder"""
        super().__init__()

        n_sin = int(dim_model / 2)
        n_cos = dim_model - n_sin

        if min_wavelength:
            base = min_wavelength / (2 * np.pi)
            scale = max_wavelength / min_wavelength
        else:
            base = 1
            scale = max_wavelength / (2 * np.pi)

        sin_term = base * scale ** (
            torch.arange(0, n_sin).float() / (n_sin - 1)
        )
        cos_term = base * scale ** (
            torch.arange(0, n_cos).float() / (n_cos - 1)
        )

        self.register_buffer("sin_term", sin_term)
        self.register_buffer("cos_term", cos_term)

    def forward(self, X):
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (n_masses)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (n_masses, dim_model)
            The encoded features for the mass spectra.
        """
        # print("sin_term shape:",self.sin_term.shape)
        # 将 X 的形状从 (n_masses,) 转换为 (n_masses, 1)，以便与 sin_term 和 cos_term 对齐
        X = X.unsqueeze(-1)
        sin_mz = torch.sin(X / self.sin_term)
        cos_mz = torch.cos(X / self.cos_term)
        
        return torch.cat([sin_mz, cos_mz], axis=-1)