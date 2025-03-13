# %%
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import dask.array as da
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from ssqueeze import _rs


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
        Tx, Sx, ssq_freqs, Sfs = _rs.ssq_stft(
            x,
            window=window,
            n_fft=n_fft,
            hop_len=hop_len,
            fs=fs,
            padtype="reflect",
            squeezing="sum",
        )

        print(f"Synchrosqueezed STFT result shape: {Tx.shape}")
        print(f"Original STFT result shape: {Sx.shape}")
        print(f"SSQ frequencies shape: {ssq_freqs.shape}")
        print(f"STFT frequencies shape: {Sfs.shape}")
        print("SSQ_STFT test successful!")
        return True
    except Exception as e:
        print(f"Error testing Rust SSQ_STFT: {e}")
        return False


# The process function that uses Dask with Synchrosqueezing
def process_ssq_stft(
    data: da.Array,
    fs: float = None,
    n_fft: int = 1024,
    hop_length: int = 256,
    window_name: str = "hann",
    squeezing: str = "sum",
    gamma: float = None,
    return_original: bool = False,  # Option to return original STFT alongside SSQ
    **kwargs,
) -> dict:
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
    squeezing : str
        Synchrosqueezing method ('sum' or 'lebesgue')
    gamma : float
        Phase threshold for synchrosqueezing
    return_original : bool
        Whether to return the original STFT alongside the synchrosqueezed version

    Returns:
    --------
    dict
        Dictionary containing:
        - 'ssq': Synchrosqueezed spectrogram with shape (freq_bins, time_frames, channels)
        - 'stft': Original STFT with shape (freq_bins, time_frames, channels) if return_original=True
        - 'ssq_freqs': Frequencies for synchrosqueezed transform
        - 'stft_freqs': Frequencies for STFT
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
        ssq_results = []
        stft_results = []
        ssq_freqs_all = None
        stft_freqs_all = None

        for ch in range(n_channels):
            # Extract channel data and ensure it's the right type
            channel_data = np.asarray(x[:, ch], dtype=np.float64)

            # Make sure window is the right type too
            window_array = np.asarray(window, dtype=np.float64)

            # Call Rust SSQ_STFT function
            try:
                Tx, Sx, ssq_freqs, Sfs = _rs.ssq_stft(
                    channel_data,
                    window=window_array,
                    n_fft=n_fft,
                    hop_len=hop_length,
                    fs=fs,
                    padtype="reflect",
                    squeezing=squeezing,
                    gamma=gamma,
                )
                # Store frequencies from the first channel
                if ch == 0:
                    ssq_freqs_all = ssq_freqs
                    stft_freqs_all = Sfs

                # Add to results
                ssq_results.append(Tx)
                if return_original:
                    stft_results.append(Sx)
            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                # Return zeros with expected shape if processing fails
                n_frames = ((x.shape[0] - n_fft) // hop_length) + 1
                n_freqs = n_fft // 2 + 1
                ssq_results.append(np.zeros((n_freqs, n_frames), dtype=np.complex128))
                if return_original:
                    stft_results.append(
                        np.zeros((n_freqs, n_frames), dtype=np.complex128)
                    )

        # Stack results, shape: (channels, freq_bins, time_frames)
        result_dict = {}

        if ssq_results:
            stacked_ssq = np.stack(ssq_results)
            # Rearrange to (freq_bins, time_frames, channels)
            result_dict["ssq"] = np.transpose(stacked_ssq, (1, 2, 0))
            result_dict["ssq_freqs"] = ssq_freqs_all

            if return_original and stft_results:
                stacked_stft = np.stack(stft_results)
                result_dict["stft"] = np.transpose(stacked_stft, (1, 2, 0))
                result_dict["stft_freqs"] = stft_freqs_all
        else:
            # In case all channels failed
            n_frames = ((x.shape[0] - n_fft) // hop_length) + 1
            n_freqs = n_fft // 2 + 1
            result_dict["ssq"] = np.zeros(
                (n_freqs, n_frames, n_channels), dtype=np.complex128
            )
            result_dict["ssq_freqs"] = np.linspace(0, fs / 2, n_freqs)

            if return_original:
                result_dict["stft"] = np.zeros(
                    (n_freqs, n_frames, n_channels), dtype=np.complex128
                )
                result_dict["stft_freqs"] = np.linspace(0, fs / 2, n_freqs)

        return result_dict

    # Use map_overlap with adjusted parameters
    result = data.map_overlap(
        process_chunk,
        depth={-2: overlap_samples},  # Using dict form like STFT
        boundary="reflect",
        dtype=np.float32,  # Return dictionary
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
result = process_ssq_stft(
    data_,
    fs=fs,
    n_fft=1024,
    hop_length=256,
    window_name="hann",
    squeezing="sum",
    return_original=True,
)
# %%
result.compute()
# %%
# Plot the results
import matplotlib.pyplot as plt

# Compute a small chunk for visualization
segment_duration = 5  # seconds
hop_length = 256
time_per_frame = hop_length / fs
frames_per_segment = int(segment_duration / time_per_frame)

# Take a slice of the data
start_frame = 0
end_frame = start_frame + frames_per_segment

# Compute just this slice
sample_result = result.compute()

# Extract the results
ssq_spectrogram = sample_result["ssq"][:, start_frame:end_frame, 0]
stft_spectrogram = sample_result["stft"][:, start_frame:end_frame, 0]
ssq_freqs = sample_result["ssq_freqs"]
stft_freqs = sample_result["stft_freqs"]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot the original STFT
stft_db = 20 * np.log10(np.abs(stft_spectrogram) + 1e-10)
vmax_stft = np.max(stft_db)
vmin_stft = vmax_stft - 80

im1 = ax1.imshow(
    stft_db,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=vmin_stft,
    vmax=vmax_stft,
)
ax1.set_title("Standard STFT Spectrogram")
ax1.set_ylabel("Frequency Bin")
fig.colorbar(im1, ax=ax1, label="Magnitude (dB)")

# Plot the synchrosqueezed STFT
ssq_db = 20 * np.log10(np.abs(ssq_spectrogram) + 1e-10)
vmax_ssq = np.max(ssq_db)
vmin_ssq = vmax_ssq - 80

im2 = ax2.imshow(
    ssq_db,
    aspect="auto",
    origin="lower",
    cmap="viridis",
    vmin=vmin_ssq,
    vmax=vmax_ssq,
)
ax2.set_title("Synchrosqueezed STFT Spectrogram")
ax2.set_xlabel("Time Frame")
ax2.set_ylabel("Frequency Bin")
fig.colorbar(im2, ax=ax2, label="Magnitude (dB)")

# Add frequency labels
for ax, freqs in [(ax1, stft_freqs), (ax2, ssq_freqs)]:
    yticks = np.linspace(0, len(freqs) - 1, 10, dtype=int)
    yticklabels = [f"{freqs[i] / 1000:.1f}" for i in yticks]  # Show in kHz
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

# Add time labels
for ax in [ax1, ax2]:
    segment_times = np.linspace(0, segment_duration, end_frame - start_frame)
    xticks = np.linspace(0, end_frame - start_frame - 1, 6, dtype=int)
    xticklabels = [f"{segment_times[i]:.1f}s" for i in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

plt.tight_layout()
plt.show()

# %%
