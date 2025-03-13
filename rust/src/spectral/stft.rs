// rust/src/spectral/stft.rs
use ndarray::{Array1, Array2, ArrayView1, s, Axis};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;
use rustfft::FftPlanner;
use std::sync::Arc;

/// Create and apply a window function
fn apply_window(signal: &ArrayView1<f64>, window: &ArrayView1<f64>) -> Array1<Complex64> {
    let n = signal.len().min(window.len());
    let mut windowed = Array1::zeros(n);
    
    for i in 0..n {
        windowed[i] = Complex64::new(signal[i] * window[i], 0.0);
    }
    
    windowed
}

/// Compute the short-time Fourier transform
#[pyfunction]
pub fn stft<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: PyReadonlyArray1<f64>,
    padtype: &str,
) -> PyResult<(PyObject, PyObject)> {
    // Convert Python arrays to Rust arrays
    let x_array = x.as_array();
    let window_array = window.as_array();
    
    // Pad the signal if needed
    let padded_x = match padtype {
        "reflect" => pad_reflect(&x_array, n_fft),
        "zero" => pad_zeros(&x_array, n_fft),
        _ => pad_reflect(&x_array, n_fft), // Default to reflect
    };
    
    // Calculate the number of frames
    let n_samples = padded_x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    // Initialize output arrays
    let mut stft_result = Array2::<Complex64>::zeros((n_freqs, n_frames));
    let freqs = Array1::<f64>::linspace(0.0, 0.5, n_freqs);
    
    // Create FFT planner for reuse
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    // Analyze each frame
    (0..n_frames).into_par_iter().for_each(|frame| {
        // Extract the frame
        let start = frame * hop_length;
        let end = start + n_fft;
        let frame_slice = padded_x.slice(s![start..end]);
        
        // Apply window
        let mut windowed_frame = apply_window(&frame_slice, &window_array);
        
        // Perform FFT
        fft.process(&mut windowed_frame.as_slice_mut().unwrap());
        
        // Store the positive frequencies (DC to Nyquist)
        let mut frame_result = stft_result.slice_mut(s![.., frame]);
        for (i, &value) in windowed_frame.slice(s![..n_freqs]).iter().enumerate() {
            frame_result[i] = value;
        }
    });

    // Convert back to Python objects
    let py_stft = stft_result.into_pyarray(py).to_object(py);
    let py_freqs = freqs.into_pyarray(py).to_object(py);
    
    Ok((py_stft, py_freqs))
}

/// Pad signal with reflection at boundaries
fn pad_reflect(x: &ArrayView1<f64>, n_fft: usize) -> Array1<f64> {
    let n = x.len();
    let pad_size = n_fft - 1; // Total padding size
    let pad_left = pad_size / 2;
    let pad_right = pad_size - pad_left;
    
    let mut padded = Array1::<f64>::zeros(n + pad_size);
    
    // Copy original signal to the middle
    for i in 0..n {
        padded[i + pad_left] = x[i];
    }
    
    // Reflect on the left
    for i in 0..pad_left {
        let mirror_idx = pad_left - i;
        if mirror_idx < n {
            padded[i] = x[mirror_idx];
        }
    }
    
    // Reflect on the right
    for i in 0..pad_right {
        let mirror_idx = n - 2 - i;
        if mirror_idx >= 0 && mirror_idx < n {
            padded[n + pad_left + i] = x[mirror_idx];
        }
    }
    
    padded
}

/// Pad signal with zeros
fn pad_zeros(x: &ArrayView1<f64>, n_fft: usize) -> Array1<f64> {
    let n = x.len();
    let pad_size = n_fft - 1;
    let pad_left = pad_size / 2;
    
    let mut padded = Array1::<f64>::zeros(n + pad_size);
    
    // Copy original signal to the middle
    for i in 0..n {
        padded[i + pad_left] = x[i];
    }
    
    padded
}

/// Compute the derivative of the STFT
fn stft_derivative(
    x: &ArrayView1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: &ArrayView1<f64>,
    diff_window: &ArrayView1<f64>,
) -> Array2<Complex64> {
    // Similar to stft but using the derivative of the window
    let n_samples = x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    let mut d_stft = Array2::<Complex64>::zeros((n_freqs, n_frames));
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    for frame in 0..n_frames {
        let start = frame * hop_length;
        let frame_slice = x.slice(s![start..start + n_fft]);
        
        // Apply derivative window
        let mut windowed_frame = apply_window(&frame_slice, &diff_window);
        
        // Perform FFT
        fft.process(&mut windowed_frame.as_slice_mut().unwrap());
        
        // Store positive frequencies
        let mut frame_result = d_stft.slice_mut(s![.., frame]);
        for (i, &value) in windowed_frame.slice(s![..n_freqs]).iter().enumerate() {
            frame_result[i] = value;
        }
    }
    
    d_stft
}

/// Compute the synchrosqueezed STFT
#[pyfunction]
pub fn ssq_stft<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: PyReadonlyArray1<f64>,
    fs: Option<f64>,
    squeezing: &str,
    gamma: Option<f64>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    // Convert Python arrays to Rust
    let x_array = x.as_array();
    let window_array = window.as_array();
    
    // Set defaults
    let fs = fs.unwrap_or(1.0);
    let gamma = gamma.unwrap_or(1e-8);
    
    // Compute STFT
    let stft_result = compute_stft(&x_array, n_fft, hop_length, &window_array, "reflect");
    
    // Compute derivative window for phase transform
    let diff_window = compute_diff_window(&window_array);
    
    // Compute STFT derivative for phase transform
    let d_stft = stft_derivative(&x_array, n_fft, hop_length, &window_array, &diff_window);
    
    // Compute instantaneous frequency
    let (inst_freqs, sfs) = compute_inst_frequency(&stft_result, &d_stft, n_fft, fs, gamma);
    
    // Perform synchrosqueezing
    let ssq_result = match squeezing {
        "sum" => synchrosqueeze_sum(&stft_result, &inst_freqs, &sfs),
        "lebesgue" => synchrosqueeze_lebesgue(&stft_result, &inst_freqs, &sfs),
        _ => synchrosqueeze_sum(&stft_result, &inst_freqs, &sfs), // Default to sum
    };
    
    // Convert back to Python objects
    let py_ssq = ssq_result.into_pyarray(py).to_object(py);
    let py_stft = stft_result.into_pyarray(py).to_object(py);
    let py_sfs = sfs.into_pyarray(py).to_object(py);
    let py_inst_freqs = inst_freqs.into_pyarray(py).to_object(py);
    
    Ok((py_ssq, py_stft, py_sfs, py_inst_freqs))
}

// Helper function to compute full STFT (internal use)
fn compute_stft(
    x: &ArrayView1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: &ArrayView1<f64>,
    padtype: &str,
) -> Array2<Complex64> {
    // Pad the signal
    let padded_x = match padtype {
        "reflect" => pad_reflect(x, n_fft),
        "zero" => pad_zeros(x, n_fft),
        _ => pad_reflect(x, n_fft),
    };
    
    let n_samples = padded_x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    let mut stft_result = Array2::<Complex64>::zeros((n_freqs, n_frames));
    
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(n_fft);
    
    for frame in 0..n_frames {
        let start = frame * hop_length;
        let frame_slice = padded_x.slice(s![start..start + n_fft]);
        
        // Apply window
        let mut windowed_frame = apply_window(&frame_slice, window);
        
        // Perform FFT
        fft.process(&mut windowed_frame.as_slice_mut().unwrap());
        
        // Store positive frequencies
        let mut frame_result = stft_result.slice_mut(s![.., frame]);
        for (i, &value) in windowed_frame.slice(s![..n_freqs]).iter().enumerate() {
            frame_result[i] = value;
        }
    }
    
    stft_result
}

// Compute derivative window (for phase transform)
fn compute_diff_window(window: &ArrayView1<f64>) -> Array1<f64> {
    let n = window.len();
    let mut diff_window = Array1::<f64>::zeros(n);
    
    // Simple central difference for interior points
    for i in 1..n-1 {
        diff_window[i] = (window[i+1] - window[i-1]) / 2.0;
    }
    
    // Forward/backward difference for endpoints
    diff_window[0] = window[1] - window[0];
    diff_window[n-1] = window[n-1] - window[n-2];
    
    diff_window
}

// Compute instantaneous frequency from phase transform
fn compute_inst_frequency(
    stft: &Array2<Complex64>,
    d_stft: &Array2<Complex64>,
    n_fft: usize,
    fs: f64,
    gamma: f64,
) -> (Array2<f64>, Array1<f64>) {
    let (n_freqs, n_frames) = stft.dim();
    let mut inst_freqs = Array2::<f64>::zeros((n_freqs, n_frames));
    
    // Frequency bins (normalized to [0, 0.5])
    let sfs = Array1::<f64>::linspace(0.0, 0.5 * fs, n_freqs);
    
    // Phase transform calculation
    for i in 0..n_freqs {
        for j in 0..n_frames {
            if stft[[i, j]].norm() > gamma {
                // Calculate instant frequency: freq bin + phase transform
                let phase_derivative = (d_stft[[i, j]] * stft[[i, j]].conj()).im / 
                                       (2.0 * std::f64::consts::PI * stft[[i, j]].norm_sqr());
                
                // Real frequency is frequency bin plus the phase transform term
                inst_freqs[[i, j]] = sfs[i] - phase_derivative * fs;
                
                // Handle potential instability in phase transform
                if !inst_freqs[[i, j]].is_finite() || inst_freqs[[i, j]] < 0.0 {
                    inst_freqs[[i, j]] = sfs[i];
                }
            } else {
                // For small magnitudes, just use the bin frequency
                inst_freqs[[i, j]] = sfs[i];
            }
        }
    }
    
    (inst_freqs, sfs)
}

// Synchrosqueezing using sum method
fn synchrosqueeze_sum(
    stft: &Array2<Complex64>,
    inst_freqs: &Array2<f64>,
    sfs: &Array1<f64>,
) -> Array2<Complex64> {
    let (n_freqs, n_frames) = stft.dim();
    let mut ssq = Array2::<Complex64>::zeros((n_freqs, n_frames));
    
    // Frequency resolution
    let freq_res = sfs[1] - sfs[0];
    
    for i in 0..n_freqs {
        for j in 0..n_frames {
            // Find closest frequency bin for the instantaneous frequency
            let inst_freq = inst_freqs[[i, j]];
            let idx = ((inst_freq - sfs[0]) / freq_res).round() as usize;
            
            // Only add if the target index is within range
            if idx < n_freqs {
                ssq[[idx, j]] += stft[[i, j]];
            }
        }
    }
    
    ssq
}

// Synchrosqueezing using Lebesgue method
fn synchrosqueeze_lebesgue(
    stft: &Array2<Complex64>,
    inst_freqs: &Array2<f64>,
    sfs: &Array1<f64>,
) -> Array2<Complex64> {
    let (n_freqs, n_frames) = stft.dim();
    let mut ssq = Array2::<Complex64>::zeros((n_freqs, n_frames));
    let n_total = (n_freqs * n_frames) as f64;
    
    // Frequency resolution
    let freq_res = sfs[1] - sfs[0];
    let unit_energy = Complex64::new(1.0 / n_total, 0.0);
    
    for i in 0..n_freqs {
        for j in 0..n_frames {
            // Find closest frequency bin for the instantaneous frequency
            let inst_freq = inst_freqs[[i, j]];
            let idx = ((inst_freq - sfs[0]) / freq_res).round() as usize;
            
            // Only add if the target index is within range
            if idx < n_freqs {
                ssq[[idx, j]] += unit_energy;
            }
        }
    }
    
    ssq
}