# Example showing how to use this with the same pattern as the provided code
# %%
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import dask.array as da
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ssqueeze import _rs


# %%
def test_rust_ssq_cwt():
    # Create a simple test signal
    fs = 1000  # 1 kHz sampling rate
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 second
    x = np.sin(2 * np.pi * 100 * t)  # 100 Hz sine wave

    # Set CWT parameters
    wavelet = "gmw"  # Generalized Morse Wavelet (default)
    nv = 16  # Voices per octave

    # Create scales logarithmically spaced
    num_scales = 32
    scales = np.logspace(1, 5, num_scales) / fs

    # Call the Rust SSQ_CWT function
    try:
        print("Calling Rust SSQ_CWT function...")
        Tx, ssq_freqs = _rs.ssq_cwt(
            x,
            wavelet=wavelet,
            scales=scales,
            fs=fs,
            nv=nv,
            padtype="reflect",
            squeezing="sum",
            maprange="peak",
        )

        print(f"SSQ_CWT output shape: {Tx.shape}")
        print(f"SSQ frequencies shape: {ssq_freqs.shape}")

        # Check if shapes make sense
        expected_freqs = scales.shape[0]  # Same number of frequency bins as scales
        expected_time = x.shape[0]

        if Tx.shape == (expected_freqs, expected_time):
            print("SSQ_CWT output has correct shape")
        else:
            print(
                f"SSQ_CWT output shape incorrect: expected {(expected_freqs, expected_time)}"
            )

        print("SSQ_CWT test successful!")
        return True
    except Exception as e:
        print(f"Error testing Rust SSQ_CWT: {e}")
        return False


def process_ssq_cwt(
    data: da.Array,
    fs: float = None,
    wavelet: str = "gmw",
    scales: Optional[np.ndarray] = None,
    nv: int = 32,
    padtype: str = "reflect",
    squeezing: str = "sum",
    maprange: str = "peak",
    **kwargs,
) -> Tuple[da.Array, np.ndarray]:
    """
    Process data with Rust-based SSQ_CWT implementation

    Parameters:
    -----------
    data : da.Array
        Input array, should be 1D or 2D (samples, channels)
    fs : float
        Sampling frequency in Hz
    wavelet : str
        Wavelet type: 'gmw' or 'morlet'
    scales : np.ndarray, optional
        Scales for the CWT. If None, logarithmically spaced scales will be used.
    nv : int
        Number of voices per octave for automatic scale generation
    padtype : str
        Padding scheme to apply on input ('reflect', 'zero', or None)
    squeezing : str
        Squeezing method ('sum' or 'lebesgue')
    maprange : str
        Frequency mapping range ('maximal', 'peak', 'energy')
    **kwargs : dict
        Additional parameters to pass to the SSQ_CWT function

    Returns:
    --------
    Tuple[da.Array, np.ndarray]
        - SSQ_CWT coefficients with shape (frequencies, time_frames, channels)
        - Associated frequencies array
    """
    if fs is None:
        raise ValueError("Sampling frequency (fs) must be provided")

    # Ensure input is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

    # Calculate overlap size for Dask - for SSQ_CWT, we need more overlap
    # We're using a conservative estimate based on the largest scale
    overlap_samples = max(
        1024, data.shape[0] // 10
    )  # At least 1024 samples or 10% of data

    # Store frequencies
    ssq_freqs_result = None

    def process_chunk(x: np.ndarray) -> np.ndarray:
        nonlocal ssq_freqs_result  # Make sure we can modify this from within

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

            # Call Rust SSQ_CWT function
            try:
                Tx, ssq_freqs = _rs.ssq_cwt(
                    channel_data,
                    wavelet=wavelet,
                    scales=scales,
                    fs=fs,
                    nv=nv,
                    padtype=padtype,
                    squeezing=squeezing,
                    maprange=maprange,
                    **kwargs,
                )

                # Store frequencies (should be the same for all channels)
                if ssq_freqs_result is None:
                    ssq_freqs_result = ssq_freqs

                # Add to results
                results.append(Tx)

            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                # Return zeros with expected shape if processing fails
                n_time = x.shape[0]
                n_freqs = (
                    100 if scales is None else len(scales)
                )  # Estimate if scales is None
                results.append(np.zeros((n_freqs, n_time), dtype=np.complex128))

        # Stack results, shape: (channels, frequencies, time_frames)
        if results:
            stacked = np.stack(results)
            # Rearrange to (frequencies, time_frames, channels)
            return np.transpose(stacked, (1, 2, 0))
        else:
            # In case all channels failed
            n_time = x.shape[0]
            n_freqs = (
                100 if scales is None else len(scales)
            )  # Estimate if scales is None
            return np.zeros((n_freqs, n_time, n_channels), dtype=np.complex128)

    # Use map_overlap with adjusted parameters
    result = data.map_overlap(
        process_chunk,
        depth={-2: overlap_samples},  # Using dict form for overlap on samples dimension
        boundary=padtype,
        dtype=np.complex128,
        new_axis=-3,  # Add frequencies dimension
    )

    # Return the result and the frequencies
    return result, ssq_freqs_result


def plot_ssq_cwt_spectrogram(
    result,
    freqs,
    fs,
    start_time=0,
    duration=None,
    channel_idx=0,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Plot the SSQ_CWT spectrogram with advanced visualization features.

    Parameters:
    -----------
    result : numpy.ndarray
        Pre-computed SSQ_CWT output with shape (frequencies, time_frames, channels)
    freqs : np.ndarray
        Frequencies array associated with the SSQ_CWT rows (in Hz)
    fs : float
        Sampling frequency in Hz
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
    """
    # Extract the channel
    if result.ndim == 3:
        spectrogram = result[:, :, channel_idx]
    else:
        spectrogram = result

    # Determine sample rate for plotting
    dt = 1.0 / fs
    total_time = spectrogram.shape[1] * dt

    # Determine start and end frames
    n_samples = spectrogram.shape[1]
    start_sample = int(start_time * fs)
    if duration is None:
        end_sample = n_samples
    else:
        end_sample = min(start_sample + int(duration * fs), n_samples)

    # Extract spectrogram segment for the time range
    if start_sample > 0 or end_sample < n_samples:
        spectrogram = spectrogram[:, start_sample:end_sample]

    # Make sure frequencies are available
    if freqs is None or len(freqs) != spectrogram.shape[0]:
        raise ValueError(
            f"Frequencies must match spectrogram rows. "
            f"Got {len(freqs) if freqs is not None else 0} frequencies for "
            f"{spectrogram.shape[0]} spectrogram rows."
        )

    # Calculate magnitude in decibels
    if np.iscomplex(spectrogram).any():
        # For complex values, take the magnitude
        magnitude = np.abs(spectrogram)
    else:
        # Already real values
        magnitude = spectrogram

    # Convert to dB
    spec_db = 20 * np.log10(magnitude + 1e-10)

    # Set vmin and vmax with a safety check
    if vmax is None:
        vmax = np.max(spec_db)
    if vmin is None:
        vmin = vmax - 80  # Show up to 80dB below the max

    # Create time vector for plotting
    times = np.linspace(
        start_time,
        start_time + (end_sample - start_sample) * dt,
        end_sample - start_sample,
    )

    # Create the plot
    plt.figure(figsize=(15, 8))

    # Create a meshgrid for pcolormesh
    time_mesh, freq_mesh = np.meshgrid(times, freqs)

    # Use pcolormesh for better frequency axis handling
    plt.pcolormesh(
        time_mesh, freq_mesh, spec_db, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto"
    )

    plt.colorbar(label="Magnitude (dB)")
    plt.title(f"SSQ_CWT Spectrogram - Channel {channel_idx + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.yscale("log")  # Use log scale for frequency axis
    plt.tight_layout()
    plt.show()


def generate_ssq_freqs(fs, nv=32, n_scales=255, scales=None, maprange="peak"):
    """
    Generate the frequencies for the synchrosqueezed transform without running cwt.

    Parameters:
    -----------
    fs : float
        Sampling frequency in Hz
    nv : int
        Number of voices per octave
    n_scales : int
        Number of scales/frequency bins to generate (ignored if scales is provided)
    scales : np.ndarray, optional
        Scales to use for frequency generation
    maprange : str
        Frequency mapping range ('maximal', 'peak', 'energy')

    Returns:
    --------
    np.ndarray
        Array of frequencies in Hz
    """
    # Determine frequency range based on maprange
    if maprange == "maximal":
        min_freq = 1.0  # Usually 1 Hz or lower
        max_freq = fs / 2  # Nyquist frequency
    else:  # "peak" or "energy" or others
        min_freq = fs * 0.002  # Approximating the lowest frequency
        max_freq = fs * 0.5  # Nyquist frequency

    # If scales are provided, use their length
    if scales is not None:
        n_scales = len(scales)

    # Generate log-spaced frequencies
    freqs = np.logspace(np.log10(min_freq), np.log10(max_freq), n_scales)

    return freqs


# %%
test_rust_ssq_cwt()

# %%
# Path to your data
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
    data_ = da.from_array(data_array, chunks=(1000, -1))
# %%
# Process with SSQ_CWT
result, freqs = process_ssq_cwt(
    data_[:100000, :], fs=fs, wavelet="gmw", nv=32, squeezing="sum", maprange="peak"
)
computed_result = result.compute()


# %%
# Plot the result
freqs = generate_ssq_freqs(fs, nv=32, n_scales=255)
print(f"Generated frequencies range: {freqs[0]:.2f} Hz to {freqs[-1]:.2f} Hz")

# Then plot the pre-computed result
plot_ssq_cwt_spectrogram(computed_result, freqs, fs=fs, channel_idx=1, duration=1)

# %%
# Make sure to extract a single 1D array with the right data type
sample_data = np.ascontiguousarray(data_[:100000, 0].compute(), dtype=np.float64)

# Verify the shape and data type
print(f"Sample data shape: {sample_data.shape}")
print(f"Sample data type: {sample_data.dtype}")

# Now try the cwt call
output = _rs.cwt(
    sample_data,  # Now a 1D array of the right type
    wavelet="gmw",
    scales=None,
    fs=fs,
    nv=32,
)

# Plot the CWT magnitude
plt.figure(figsize=(15, 8))
plt.imshow(np.abs(output[0]), aspect="auto", origin="lower")
plt.colorbar(label="Magnitude")
plt.title("CWT Magnitude")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.show()

# %%
# Try with a different squeezing method and higher gamma value
Tx, ssq_freqs = _rs.ssq_cwt(
    sample_data,
    wavelet="gmw",
    scales=None,
    fs=fs,
    nv=32,
    squeezing="sum",
    maprange="maximal",
    gamma=1e-6,  # Increase the threshold
)

# Plot the CWT magnitude
plt.figure(figsize=(15, 8))
plt.imshow(np.abs(Tx[300:, :]), aspect="auto", origin="lower")
plt.colorbar(label="Magnitude")
plt.title("SSQ CWT Magnitude")
plt.xlabel("Time")
plt.ylabel("Scale")
plt.show()
# %%
