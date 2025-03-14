"""
Type hint stub file for the _rs Rust extension module.
"""

from typing import List, Optional, Tuple, Union, overload

import numpy as np

def hello_from_bin() -> str: ...
@overload
def stft(
    x: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray, padtype: str
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def cwt(
    x: np.ndarray,
    wavelet: str = "gmw",
    scales: Optional[np.ndarray] = None,
    fs: Optional[float] = None,
    t: Optional[np.ndarray] = None,
    nv: int = 32,
    l1_norm: bool = True,
    derivative: bool = False,
    padtype: str = "reflect",
    rpadded: bool = False,
    vectorized: bool = True,
    patience: int = 0,
) -> Union[
    Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
]: ...
@overload
def icwt(
    Wx: np.ndarray,
    wavelet: str = "gmw",
    scales: Optional[np.ndarray] = None,
    nv: Optional[int] = None,
    one_int: bool = True,
    x_len: Optional[int] = None,
    x_mean: float = 0,
    padtype: str = "reflect",
    rpadded: bool = False,
    l1_norm: bool = True,
) -> np.ndarray: ...
@overload
def ssq_stft(
    x: np.ndarray,
    window: np.ndarray,
    n_fft: Optional[int] = None,
    win_len: Optional[int] = None,
    hop_len: int = 1,
    fs: float = 1.0,
    padtype: str = "reflect",
    squeezing: str = "sum",
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]: ...
@overload
def morlet(w: np.ndarray, mu: float = 6.0, dtype: str = "float64") -> np.ndarray: ...
@overload
def morlet_freq(
    n: int = 1024, scale: float = 1.0, mu: float = 6.0, dtype: str = "float64"
) -> np.ndarray: ...
@overload
def morlet_time(
    n: int = 1024, scale: float = 1.0, mu: float = 6.0, dtype: str = "float64"
) -> np.ndarray: ...
@overload
def gmw(
    w: np.ndarray,
    gamma: float = 3.0,
    beta: float = 60.0,
    norm: str = "bandpass",
    order: int = 0,
    dtype: str = "float64",
) -> np.ndarray: ...
@overload
def gmw_freq(
    n: int = 1024,
    scale: float = 1.0,
    gamma: float = 3.0,
    beta: float = 60.0,
    norm: str = "bandpass",
    order: int = 0,
    dtype: str = "float64",
) -> np.ndarray: ...
@overload
def gmw_time(
    n: int = 1024,
    scale: float = 1.0,
    gamma: float = 3.0,
    beta: float = 60.0,
    norm: str = "bandpass",
    order: int = 0,
    dtype: str = "float64",
) -> np.ndarray: ...
@overload
def gmw_center_frequency(
    gamma: float = 3.0, beta: float = 60.0, kind: str = "peak"
) -> float: ...
