from typing import Any, Tuple

import numpy as np

def hello_from_bin() -> str: ...
def stft(
    x: np.ndarray, n_fft: int, hop_length: int, window: np.ndarray, padtype: str
) -> Tuple[np.ndarray, np.ndarray]: ...
