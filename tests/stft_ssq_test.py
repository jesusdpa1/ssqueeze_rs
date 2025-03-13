# %%
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import dask.array as da
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ssqueeze import _rs


def plot_ssq_stft(
    ssq_stft_result,
    fs,
    hop_length=256,
    start_time=0,
    duration=None,
    channel_idx=0,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Plot the synchrosqueezed STFT spectrogram.

    Parameters:
    -----------
    ssq_stft_result : dask.Array
        Synchrosqueezed STFT output from process_stft_ssq()
    fs : float
        Sampling frequency
    n_fft : int, optional
        FFT size used in processing
    hop_length : int, optional
        Hop length used in STFT processing
    start_time : float, optional
        Starting time of the segment to plot (in seconds)
    duration : float, optional
        Duration of the segment to plot (in seconds).
        If None, plots the entire spectrogram
    channel_idx : int, optional
        Channel index to plot (for multichannel data)
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    cmap : str, optional
        Colormap to use for the spectrogram
    """
    # Compute the result (if it's a Dask array)
    try:
        import dask.array as da

        if isinstance(ssq_stft_result, da.Array):
            ssq_stft_result = ssq_stft_result.compute()
    except ImportError:
        pass

    # Extract the channel
    if ssq_stft_result.ndim == 3:
        ssq_stft_result = ssq_stft_result[:, :, channel_idx]

    # Compute time per frame
    time_per_frame = hop_length / fs

    # Compute frequency axis
    freq_bins = ssq_stft_result.shape[0]
    freq_axis = np.linspace(0, fs / 2, freq_bins)

    # Determine start and end frames
    total_frames = ssq_stft_result.shape[1]
    start_frame = int(start_time / time_per_frame)
    if duration is None:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + int(duration / time_per_frame), total_frames)

    # Extract spectrogram segment
    ssq_segment = ssq_stft_result[:, start_frame:end_frame]

    # Compute magnitude in decibels
    # Add a small epsilon to avoid log(0)
    eps = 1e-10
    ssq_db = 20 * np.log10(np.abs(ssq_segment) + eps)

    # Determine color scaling
    if vmin is None or vmax is None:
        ssq_finite = ssq_db[np.isfinite(ssq_db)]
        if len(ssq_finite) > 0:
            if vmax is None:
                vmax = np.percentile(ssq_finite, 99)
            if vmin is None:
                vmin = vmax - 80  # Show up to 80dB below the max

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Use symmetric normalization to handle potential large dynamic range
    norm = colors.SymLogNorm(linthresh=1, linscale=1, vmin=vmin, vmax=vmax)

    plt.imshow(
        ssq_db,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[
            start_frame * time_per_frame,
            end_frame * time_per_frame,
            freq_axis[0],
            freq_axis[-1],
        ],
    )

    plt.colorbar(label="Magnitude (dB)")

    plt.title(f"Synchrosqueezed STFT - Channel {channel_idx + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


# Create a test function to verify the Rust module
def test_rust_ssq_stft():
    # Create a simple test signal
    fs = 1000  # 1 kHz sampling rate
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second
    x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

    # Set STFT parameters
    n_fft = 256
    hop_len = 64
    window = np.hanning(n_fft)

    # Call the Rust Synchrosqueezed STFT function
    try:
        print("Calling Rust SSQ_STFT function...")
        Tx, ssq_freqs = _rs.ssq_stft(
            x,
            window=window,
            n_fft=n_fft,
            hop_len=hop_len,
            fs=fs,
            padtype="reflect",
            squeezing="sum",
        )

        print(f"Synchrosqueezed STFT result shape: {Tx.shape}")
        print(f"SSQ frequencies shape: {ssq_freqs.shape}")
        print("SSQ_STFT test successful!")
        return True
    except Exception as e:
        print(f"Error testing Rust SSQ_STFT: {e}")
        return False


def process_stft_ssq(
    data: da.Array,
    fs: float = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    window_name: str = "hann",
    squeezing: str = "sum",
    **kwargs,
) -> da.Array:
    """
    Process data with Rust-based Synchrosqueezed STFT implementation

    Parameters:
    -----------
    data : da.Array
        Input array, should be 1D or 2D (samples, channels)
    fs : float
        Sampling frequency in Hz
    n_fft : int
        FFT size
    hop_length : int
        Hop length between consecutive frames
    window_name : str
        Window function name
    squeezing : str, optional
        Synchrosqueezing method ('sum' or 'lebesgue')

    Returns:
    --------
    da.Array
        Synchrosqueezed spectrogram with shape (freq_bins, time_frames, channels)
    """
    if fs is None:
        raise ValueError("Sampling frequency (fs) must be provided")

    # Ensure input is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

    # Create window function
    if window_name == "hann":
        window = np.hanning(n_fft)
    elif window_name == "hamming":
        window = np.hamming(n_fft)
    elif window_name == "blackman":
        window = np.blackman(n_fft)
    else:
        # Default to Hann window
        window = np.hanning(n_fft)

    # Calculate overlap size for Dask
    overlap_samples = n_fft  # Be conservative with overlap for STFT

    def process_chunk(x: np.ndarray) -> np.ndarray:
        # Ensure the input is a contiguous array
        x = np.ascontiguousarray(x)

        # Handle single-channel case
        if x.ndim == 1:
            x = x.reshape(-1, 1)

        n_channels = x.shape[1]
        results = []
        results_freqs = []

        for ch in range(n_channels):
            # Extract channel data and ensure it's the right type
            channel_data = np.asarray(x[:, ch], dtype=np.float64)

            # Make sure window is the right type too
            window_array = np.asarray(window, dtype=np.float64)

            # Call Rust Synchrosqueezed STFT function
            try:
                ssq_stft_result, ssq_freqs = _rs.ssq_stft(
                    channel_data,
                    window_array,
                    n_fft=n_fft,
                    win_len=n_fft,
                    hop_len=hop_length,
                    fs=fs,
                    padtype="reflect",
                    squeezing=squeezing,
                )
                # Add to results
                results.append(ssq_stft_result)
                results_freqs.append(ssq_freqs)
            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                # Return zeros with expected shape if processing fails
                n_frames = ((x.shape[0] - n_fft) // hop_length) + 1
                n_freqs = n_fft // 2 + 1
                results.append(np.zeros((n_freqs, n_frames), dtype=np.complex128))
                results_freqs.append(np.linspace(0, fs / 2, n_freqs, dtype=np.float64))

        # Ensure consistent frequency grid
        frequencies = results_freqs[0]  # All channels should have same freq grid

        # Stack results, shape: (channels, freq_bins, time_frames)
        if results:
            stacked = np.stack(results)
            # Rearrange to (freq_bins, time_frames, channels)
            return np.transpose(stacked, (1, 2, 0))
        else:
            # In case all channels failed
            n_frames = ((x.shape[0] - n_fft) // hop_length) + 1
            n_freqs = n_fft // 2 + 1
            return np.zeros((n_freqs, n_frames, n_channels), dtype=np.complex128)

    # Use map_overlap with adjusted parameters
    result = data.map_overlap(
        process_chunk,
        depth={-2: overlap_samples},  # Using dict form like STFT
        boundary="reflect",
        dtype=np.complex128,
        new_axis=-3,  # Add frequency bin dimension
    )

    return result


# %%
test_success = test_rust_ssq_stft()
# %%

path_home = Path(r"E:\jpenalozaa")
path_base = path_home.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline\RawG.ant"
)

parquet_path = path_base.joinpath("data_RawG_t0_0-300_0.parquet")
fs = 24414.0625

# Load the data
with pa.memory_map(str(parquet_path), "r") as mmap:
    table = pq.read_table(mmap)
    data_array = table.to_pandas().values
    data_ = da.from_array(data_array, chunks=(1000000, -1))
# %%
# Process with synchrosqueezed STFT
result = process_stft_ssq(
    data_,
    fs=fs,
    n_fft=1024,
    hop_length=256,
    window_name="hann",
    squeezing="sum",
)
# %%
a = result.compute()


# %%
plot_ssq_stft(result, fs=fs, hop_length=256, duration=1)

# %%
