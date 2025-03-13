from typing import Any, Optional, Tuple

import numpy as np

def hello_from_bin() -> str: ...
def stft(
    x: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray, padtype: str
) -> Tuple[np.ndarray, np.ndarray]: ...
def ssq_stft(
    x: np.ndarray,
    window: np.ndarray,
    n_fft: Optional[int] = None,
    win_len: Optional[int] = None,
    hop_len: Optional[int] = None,
    fs: Optional[float] = None,
    padtype: Optional[str] = None,
    squeezing: Optional[str] = None,
    gamma: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...
