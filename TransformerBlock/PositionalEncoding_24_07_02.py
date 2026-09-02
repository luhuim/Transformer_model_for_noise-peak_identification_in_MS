# /usr/bin/python3
#G:\Lund_master_thesis\pyteomics\PositionalEncoding_24_07_02.py
#%% 导入一些前置的class
"""借鉴CasaNovo的代码，制作我的encoder. Tranformer models to handle mass spectra."""
from collections.abc import Callable
import torch
import math
# import einops
import numpy as np

#%%导入一些前置class
"""Helpful Mixins."""
class ModelMixin:
    """Add helpful properties for depthcharge models."""

    @property
    def device(self) -> torch.device:
        """The current device for first parameter of the model."""
        return next(self.parameters()).device
class TransformerMixin:
    """Properties shared by Transformer models."""

    @property
    def d_model(self) -> int:
        """The latent dimensionality of the model."""
        return self._d_model

    @property
    def nhead(self) -> int:
        """The number of attention heads."""
        return self._nhead

    @property
    def dim_feedforward(self) -> int:
        """The dimensionality of the Transformer feedforward layers."""
        return self._dim_feedforward

    @property
    def n_layers(self) -> int:
        """The number of Transformer layers."""
        return self._n_layers

    @property
    def dropout(self) -> float:
        """The dropout for the transformer layers."""
        return self._dropout
#%% Positional Encoding的sub-class,位置编码 spectrum
"""Simple encoders for input into Transformers and the like."""
class FloatEncoder(torch.nn.Module):
    """Encode floating point values using sine and cosine waves.
    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_wavelength : float, optional
        The minimum wavelength to use.
    max_wavelength : float, optional
        The maximum wavelength to use.
    learnable_wavelengths : bool, optional
        Allow the selected wavelengths to be fine-tuned
        by the model.
    """

    def __init__(
        self,
        d_model: int,
        min_wavelength: float = 0.001,
        max_wavelength: float = 10000,
        learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MassEncoder."""
        super().__init__()
        # Error checking:
        if min_wavelength <= 0:
            raise ValueError("'min_wavelength' must be greater than 0.")

        if max_wavelength <= 0:
            raise ValueError("'max_wavelength' must be greater than 0.")

        self.learnable_wavelengths = learnable_wavelengths

        # Get dimensions for equations:
        d_sin = math.ceil(d_model / 2)
        d_cos = d_model - d_sin

        base = min_wavelength / (2 * np.pi)
        scale = max_wavelength / min_wavelength
        sin_exp = torch.arange(0, d_sin).float() / (d_sin - 1)
        cos_exp = (torch.arange(d_sin, d_model).float() - d_sin) / (d_cos - 1)
        sin_term = base * (scale**sin_exp)
        cos_term = base * (scale**cos_exp)

        if not self.learnable_wavelengths:
            self.register_buffer("sin_term", sin_term)
            self.register_buffer("cos_term", cos_term)
        else:
            self.sin_term = torch.nn.Parameter(sin_term)
            self.cos_term = torch.nn.Parameter(cos_term)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values.

        Parameters
        ----------
        X : torch.Tensor of shape (batch_size, n_float)
            The masses to embed.

        Returns
        -------
        torch.Tensor of shape (batch_size, n_float, d_model)
            The encoded features for the floating point numbers.
        """
        # print("----now is running FloatEncoder----")
        # print("running'sin_mz = torch.sin(X[:, :, None] / self.sin_term)'")
        # print("X[:, :, None],shape",X[:, :, None].shape)
        # print("X[:, :, None] is:\n",X[:, :, None])
        
        sin_mz = torch.sin(X[:, :, None] / self.sin_term)
        # print("'sin_mz' is:\n", sin_mz)
        # print(f"the shape of 'sin_mz' is {sin_mz.shape}")
        cos_mz = torch.cos(X[:, :, None] / self.cos_term)
        # print(f"the shape of 'cos_mz' is {sin_mz.shape}")
        # print("the output shape of this time is:",torch.cat([sin_mz, cos_mz], axis=-1).shape)
        # print("----finish----")
        return torch.cat([sin_mz, cos_mz], axis=-1)

#%% Positional Encoding的主class  位置编码 spectrum
class PeakEncoder(torch.nn.Module):
    """Encode mass spectrum.

    Parameters
    ----------
    d_model : int
        The number of features to output.
    min_mz_wavelength : float, optional
        The minimum wavelength to use for m/z.
    max_mz_wavelength : float, optional
        The maximum wavelength to use for m/z.
    min_intensity_wavelength : float, optional
        The minimum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    max_intensity_wavelength : float, optional
        The maximum wavelength to use for intensity. The default assumes
        intensities between [0, 1].
    learnable_wavelengths : bool, optional
        Allow the selected wavelengths to be fine-tuned
        by the model.

    """

    def __init__(
        self,
        d_model: int,
        min_mz_wavelength: float = 0.001,
        max_mz_wavelength: float = 10000,
        min_intensity_wavelength: float = 1e-6,
        max_intensity_wavelength: float = 1,
        learnable_wavelengths: bool = False,
    ) -> None:
        """Initialize the MzEncoder."""
        super().__init__()
        self.d_model = d_model
        self.learnable_wavelengths = learnable_wavelengths

        self.mz_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_mz_wavelength,
            max_wavelength=max_mz_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )

        self.int_encoder = FloatEncoder(
            d_model=self.d_model,
            min_wavelength=min_intensity_wavelength,
            max_wavelength=max_intensity_wavelength,
            learnable_wavelengths=learnable_wavelengths,
        )
        self.combiner = torch.nn.Linear(2 * d_model, d_model, bias=False)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Encode m/z values and intensities.

        Note that we expect intensities to fall within the interval [0, 1].

        Parameters
        ----------
        X : torch.Tensor of shape (n_spectra, n_peaks, 2)
            The spectra to embed. 
                Axis 0 represents a mass spectrum
                Axis 1 contains the peaks in the mass spectrum
                Axis 2 is essentially a 2-tuple specifying the m/z-intensity pair for each peak.
            These should be zero-padded, such that all of the spectra in the batch
            are the same length.

        Returns
        -------
        torch.Tensor of shape (n_spectra, n_peaks, d_model)
            The encoded features for the mass spectra.

        """
        # print("####Now is in PeakEncoder####")
        # print("the shape of input:\n",X.shape)
        # print("mz_encoder,input:\n",X[:, :, 0].shape)
        # print("int_encoder, input\n",X[:, :, 1].shape)
        encoded = torch.cat(
            [
                self.mz_encoder(X[:, :, 0]),
                self.int_encoder(X[:, :, 1]),
            ],
            dim=2,
        )
        # print("After this step 'torch.cat(mz_encoder(X[:, :, 0]),int_encoder(X[:, :, 1])'")
        # print("the output shape is :",encoded.shape)
        # print("####finish####")
        return self.combiner(encoded)
    

#%% encoder for mass 
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

