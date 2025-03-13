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


def plot_stft_spectrogram(
    result,
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
    Plot the STFT spectrogram with advanced visualization features.

    Parameters:
    -----------
    result : dask.Array or numpy.ndarray
        STFT output with shape (freq_bins, time_frames, channels)
    fs : float
        Sampling frequency in Hz
    hop_length : int, optional
        Hop length between STFT frames (default: 256)
    start_time : float, optional
        Starting time of the segment to plot (in seconds, default: 0)
    duration : float, optional
        Duration of the segment to plot (in seconds).
        If None, plots the entire spectrogram
    channel_idx : int, optional
        Channel index to plot (default: 0)
    vmin : float, optional
        Minimum value for color scaling
    vmax : float, optional
        Maximum value for color scaling
    cmap : str, optional
        Colormap to use for the spectrogram (default: 'viridis')

    Returns:
    --------
    None
        Displays the spectrogram plot
    """
    # Compute the result if it's a Dask array
    try:
        import dask.array as da

        if isinstance(result, da.Array):
            result = result.compute()
    except ImportError:
        pass

    # Extract the channel
    if result.ndim == 3:
        result = result[:, :, channel_idx]

    # Compute time per frame
    time_per_frame = hop_length / fs

    # Determine start and end frames
    total_frames = result.shape[1]
    start_frame = int(start_time / time_per_frame)
    if duration is None:
        end_frame = total_frames
    else:
        end_frame = min(start_frame + int(duration / time_per_frame), total_frames)

    # Extract spectrogram segment
    spectrogram = result[:, start_frame:end_frame]

    # Check if the spectrogram has valid data
    if spectrogram.size == 0 or np.max(np.abs(spectrogram)) < 1e-10:
        print("Warning: Spectrogram data is empty or contains only zeros.")
        # Create a dummy spectrogram for plotting
        spec_db = np.zeros((spectrogram.shape[0], spectrogram.shape[1]))
        vmin, vmax = -80, 0
    else:
        # Plot the magnitude spectrogram in decibels
        spec_db = 20 * np.log10(np.abs(spectrogram) + 1e-10)

        # Set vmin and vmax with a safety check
        if vmax is None:
            vmax = np.max(spec_db)
        if vmin is None:
            vmin = vmax - 80  # Show up to 80dB below the max

    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[start_frame * time_per_frame, end_frame * time_per_frame, 0, fs / 2],
    )

    plt.colorbar(label="Magnitude (dB)")
    plt.title(
        f"STFT Spectrogram - Channel {channel_idx + 1} "
        f"(segment {start_frame}-{end_frame})"
    )
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    # Add frequency labels
    freq_max = fs / 2  # Nyquist frequency
    yticks = np.linspace(0, freq_max, 10)
    plt.yticks(yticks, [f"{freq:.1f}" for freq in yticks])

    # Add time labels relative to the segment
    if end_frame > start_frame:
        segment_times = np.linspace(0, (end_frame - start_frame) * time_per_frame, 6)
        plt.xticks(segment_times, [f"{t:.1f}s" for t in segment_times])

    plt.tight_layout()
    plt.show()


# Create a test function to verify the Rust module
def test_rust_stft():
    # Create a simple test signal
    fs = 1000  # 1 kHz sampling rate
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second
    x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

    # Set STFT parameters
    n_fft = 256
    hop_length = 64
    window = np.hanning(n_fft)

    # Call the Rust STFT function
    try:
        print("Calling Rust STFT function...")
        stft_result, freqs = _rs.stft(
            x, n_fft=n_fft, hop_length=hop_length, window=window, padtype="reflect"
        )

        print(f"STFT result shape: {stft_result.shape}")
        print(f"Frequencies shape: {freqs.shape}")
        print("STFT test successful!")
        return True
    except Exception as e:
        print(f"Error testing Rust STFT: {e}")
        return False


# The process function that uses Dask
def process_stft(
    data: da.Array,
    fs: float = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    window_name: str = "hann",
    **kwargs,
) -> da.Array:
    """
    Process data with Rust-based STFT implementation

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

    Returns:
    --------
    da.Array
        Spectrogram with shape (freq_bins, time_frames, channels)
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

        for ch in range(n_channels):
            # Extract channel data and ensure it's the right type
            channel_data = np.asarray(x[:, ch], dtype=np.float64)

            # Make sure window is the right type too
            window_array = np.asarray(window, dtype=np.float64)

            # Call Rust STFT function
            try:
                stft_result, freqs = _rs.stft(
                    channel_data,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    window=window_array,
                    padtype="reflect",
                )
                # Add to results
                results.append(stft_result)
            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                # Return zeros with expected shape if processing fails
                n_frames = ((x.shape[0] - n_fft) // hop_length) + 1
                n_freqs = n_fft // 2 + 1
                results.append(np.zeros((n_freqs, n_frames), dtype=np.complex128))

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
        dtype=np.float32,
        new_axis=-3,  # Add frequency bin dimension
    )

    return result


def plot_stft_magnitude(
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
    Plot the magnitude of the STFT spectrogram.

    Parameters are the same as plot_ssq_stft.
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
    stft_segment = ssq_stft_result[:, start_frame:end_frame]

    # Compute magnitude in decibels
    # Add a small epsilon to avoid log(0)
    eps = 1e-10
    stft_db = 20 * np.log10(np.abs(stft_segment) + eps)

    # Determine color scaling
    if vmin is None or vmax is None:
        stft_finite = stft_db[np.isfinite(stft_db)]
        if len(stft_finite) > 0:
            if vmax is None:
                vmax = np.percentile(stft_finite, 99)
            if vmin is None:
                vmin = vmax - 80  # Show up to 80dB below the max

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Use symmetric normalization to handle potential large dynamic range
    norm = colors.SymLogNorm(linthresh=1, linscale=1, vmin=vmin, vmax=vmax)

    plt.imshow(
        stft_db,
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

    plt.title(f"STFT Magnitude - Channel {channel_idx + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")

    plt.tight_layout()
    plt.show()


# %%
path_home = Path(r"E:\jpenalozaa")
path_base = path_home.joinpath(
    r"topoMapping\25-02-26_9881-2_testSubject_topoMapping\drv\drv_00_baseline\RawG.ant"
)

parquet_path = path_base.joinpath("data_RawG_t0_0-300_0.parquet")
fs = 24414.0625
# %%
with pa.memory_map(str(parquet_path), "r") as mmap:
    table = pq.read_table(mmap)
    data_array = table.to_pandas().values
    data_ = da.from_array(data_array, chunks=(1000000, -1))


# %%
result = process_stft(data_, fs=fs, n_fft=1024, hop_length=256, window_name="hann")


# %%
plot_stft_spectrogram(result, fs=fs, hop_length=256, channel_idx=1, duration=1)
# %%
