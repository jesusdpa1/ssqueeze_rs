from typing import Any, Literal, Optional, Tuple, Union

import numpy as np

def hello_from_bin() -> str: ...
def stft(
    x: np.ndarray,
    n_fft: int,
    hop_length: int,
    window: np.ndarray,
    padtype: str = "reflect",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the Short-Time Fourier Transform using Rust.

    Parameters:
        x: Input signal (1D array)
        n_fft: FFT size
        hop_length: Hop size between frames
        window: Window function array
        padtype: Padding type ("reflect" or "zero")

    Returns:
        Tuple containing:
        - STFT result (complex-valued 2D array, shape: [n_freqs, n_frames])
        - Frequency bins (1D array)
    """
    ...

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
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the Synchrosqueezed Short-Time Fourier Transform using Rust.

    Parameters:
        x: Input signal (1D array)
        window: Window function array
        n_fft: FFT size (defaults to min(len(x), 512) if None)
        win_len: Window length (defaults to len(window) if None)
        hop_len: Hop size between frames
        fs: Sampling frequency
        padtype: Padding type ("reflect" or "zero")
        squeezing: Synchrosqueezing method ("sum" or "lebesgue")
        gamma: Phase transform threshold (defaults to 10*eps if None)

    Returns:
        Tuple containing:
        - Synchrosqueezed STFT (complex-valued 2D array, shape: [n_freqs, n_frames])
        - Original STFT (complex-valued 2D array, shape: [n_freqs, n_frames])
        - Synchrosqueezed frequency bins (1D array)
        - Original STFT frequency bins (1D array)
    """
    ...
