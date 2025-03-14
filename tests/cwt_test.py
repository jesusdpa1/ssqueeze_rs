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
def process_cwt(
    data: da.Array,
    fs: float = None,
    wavelet: str = "gmw",
    scales: Optional[np.ndarray] = None,
    nv: int = 32,
    derivative: bool = False,
    padtype: str = "reflect",
    **kwargs,
) -> da.Array:
    """
    Process data with Rust-based CWT implementation

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
    derivative : bool
        Whether to also compute the derivative of the CWT
    padtype : str
        Padding scheme to apply on input ('reflect', 'zero', or None)
    **kwargs : dict
        Additional parameters to pass to the CWT function

    Returns:
    --------
    da.Array
        CWT coefficients with shape (scales, time_frames, channels)
    """
    if fs is None:
        raise ValueError("Sampling frequency (fs) must be provided")

    # Ensure input is 2D
    if data.ndim == 1:
        data = data.reshape(-1, 1)
    elif data.ndim > 2:
        raise ValueError(f"Expected 1D or 2D input, got shape {data.shape}")

    # Calculate overlap size for Dask - for CWT, we need more overlap
    # We're using a conservative estimate based on the largest scale
    overlap_samples = max(
        1024, data.shape[0] // 10
    )  # At least 1024 samples or 10% of data

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

            # Call Rust CWT function
            try:
                if derivative:
                    cwt_result, scales_out, dWx = _rs.cwt(
                        channel_data,
                        wavelet=wavelet,
                        scales=scales,
                        fs=fs,
                        nv=nv,
                        derivative=True,
                        padtype=padtype,
                        **kwargs,
                    )
                    # Add to results (ignoring derivative for now)
                    results.append(cwt_result)
                else:
                    cwt_result, scales_out = _rs.cwt(
                        channel_data,
                        wavelet=wavelet,
                        scales=scales,
                        fs=fs,
                        nv=nv,
                        derivative=False,
                        padtype=padtype,
                        **kwargs,
                    )
                    # Add to results
                    results.append(cwt_result)
            except Exception as e:
                print(f"Error processing channel {ch}: {e}")
                # Return zeros with expected shape if processing fails
                n_time = x.shape[0]
                n_scales = (
                    100 if scales is None else len(scales)
                )  # Estimate if scales is None
                results.append(np.zeros((n_scales, n_time), dtype=np.complex128))

        # Stack results, shape: (channels, scales, time_frames)
        if results:
            stacked = np.stack(results)
            # Rearrange to (scales, time_frames, channels)
            return np.transpose(stacked, (1, 2, 0))
        else:
            # In case all channels failed
            n_time = x.shape[0]
            n_scales = (
                100 if scales is None else len(scales)
            )  # Estimate if scales is None
            return np.zeros((n_scales, n_time, n_channels), dtype=np.complex128)

    # Use map_overlap with adjusted parameters
    result = data.map_overlap(
        process_chunk,
        depth={-2: overlap_samples},  # Using dict form for overlap on samples dimension
        boundary=padtype,
        dtype=np.complex128,
        new_axis=-3,  # Add scales dimension
    )

    return result


def plot_cwt_spectrogram(
    result,
    fs,
    scales=None,
    start_time=0,
    duration=None,
    channel_idx=0,
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Plot the CWT spectrogram with advanced visualization features.

    Parameters:
    -----------
    result : dask.Array or numpy.ndarray
        CWT output with shape (scales, time_frames, channels)
    fs : float
        Sampling frequency in Hz
    scales : np.ndarray, optional
        Scales used in the transform. If None, will be estimated.
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

    # Determine sample rate for plotting
    dt = 1.0 / fs
    total_time = result.shape[1] * dt

    # Determine start and end frames
    n_samples = result.shape[1]
    start_sample = int(start_time * fs)
    if duration is None:
        end_sample = n_samples
    else:
        end_sample = min(start_sample + int(duration * fs), n_samples)

    # Extract spectrogram segment
    spectrogram = result[:, start_sample:end_sample]

    # Estimate scales to frequencies if not provided
    n_scales = spectrogram.shape[0]
    if scales is None:
        # Rough estimate for visualization purposes - this assumes log-spaced scales
        max_freq = fs / 2
        min_freq = 1.0  # 1 Hz as minimum
        frequencies = np.exp(np.linspace(np.log(min_freq), np.log(max_freq), n_scales))
    else:
        # Convert scales to frequencies - rough approximation based on GMW wavelet
        frequencies = fs / (2 * scales)

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

    # Create time and frequency vectors for plotting
    times = np.linspace(
        start_time,
        start_time + (end_sample - start_sample) * dt,
        end_sample - start_sample,
    )

    # Create the plot
    plt.figure(figsize=(15, 8))
    plt.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=[times[0], times[-1], 0, n_scales - 1],
    )

    # Create frequency labels on the y-axis
    n_ticks = 10
    tick_positions = np.linspace(0, n_scales - 1, n_ticks).astype(int)
    tick_labels = [f"{frequencies[i]:.1f}" for i in tick_positions]
    plt.yticks(tick_positions, tick_labels)

    plt.colorbar(label="Magnitude (dB)")
    plt.title(f"CWT Spectrogram - Channel {channel_idx + 1}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.show()


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
    data_ = da.from_array(data_array, chunks=(1000000, -1))
# %%
# Process with CWT
result = process_cwt(data_, fs=fs, wavelet="gmw", nv=32)
result.compute()
# %%
# Plot the result
plot_cwt_spectrogram(result, fs=fs, channel_idx=1, duration=1)

# %%
