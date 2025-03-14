# Synchrosqueezing Toolkit 🌊🔍

A high-performance, Rust-powered signal processing library for Python, providing advanced time-frequency analysis techniques.

## 🚀 Project Overview

This project is a Rust-backed reimplementation of [ssqueezepy](https://github.com/OverLordGoldDragon/ssqueezepy), focusing on delivering ultra-fast and memory-efficient time-frequency analysis methods.

### 🔬 Key Features

#### Implemented
- [x] Short-Time Fourier Transform (STFT)
  - Forward and inverse transforms
  - Synchrosqueezed STFT

#### Planned Ports
- [ ] Continuous Wavelet Transform (CWT)
  - Forward and inverse transforms
  - Synchrosqueezed CWT
- [ ] Generalized Morse Wavelets
- [ ] Ridge extraction
- [ ] Advanced visualization tools
- [ ] Comprehensive testing suite

## 🛠 Technical Highlights

- **Language**: Rust with Python bindings
- **Backend**: 
  - Parallel processing with Rayon
  - SIMD optimizations
  - Zero-copy data transfer
- **Performance Goals**: 
  - Memory-efficient large dataset handling
- **Development Tools**:
  - `uv` for dependency management
  - `maturin` for Rust-Python build integration
  - `dask` for distributed computing support
  - `pyarrow` for efficient data manipulation and columnar data formats
  - `numpy` for core numerical computing
  - `pyo3` for Python-Rust bindings

## 📦 Installation

```bash
# Recommended: Use pip with pre-built wheels
pip install ssqueeze

# From source
git clone https://github.com/yourusername/ssqueeze
cd ssqueeze
uv sync
```

## 🚦 Quick Start

```py
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import dask.array as da
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from ssqueeze import _rs

fs = 1000  # 1 kHz sampling rate
t = np.linspace(0, 1, fs, endpoint=False)  # 1 second
x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

# Set STFT parameters
n_fft = 256
hop_length = 64
window = np.hanning(n_fft)

stft_result, freqs = _rs.stft(
    x, n_fft=n_fft, hop_length=hop_length, window=window, padtype="reflect"
)

```


