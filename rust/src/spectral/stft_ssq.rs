// rust/src/spectral/stft_ssq.rs
use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use crate::spectral::stft_utils::{apply_window, pad_reflect, pad_zeros, stft_derivative, compute_stft};

/// Phase transform for STFT
fn phase_stft(
    Sx: &Array2<Complex64>,
    dSx: &Array2<Complex64>,
    Sfs: &Array1<f64>,
    gamma: f64,
) -> Array2<f64> {
    let (n_freqs, n_frames) = (Sx.shape()[0], Sx.shape()[1]);
    let mut w = Array2::<f64>::zeros((n_freqs, n_frames));
    
    // Compute phase transform
    for i in 0..n_freqs {
        for j in 0..n_frames {
            if Sx[[i, j]].norm() < gamma {
                w[[i, j]] = f64::INFINITY;
            } else {
                let a = dSx[[i, j]].re;
                let b = dSx[[i, j]].im;
                let c = Sx[[i, j]].re;
                let d = Sx[[i, j]].im;
                
                // Im((1/2π) * (1/Sx) * d/dt[Sx])
                let phase_derivative = (b * c - a * d) / ((c * c + d * d) * 6.283185307179586);
                w[[i, j]] = (Sfs[i] - phase_derivative).abs();
            }
        }
    }
    
    w
}

/// Helper function to compute associated frequencies for synchrosqueezing
fn compute_associated_frequencies(
    n_freqs: usize,
    fs: f64,
    dtype: &str,
) -> Array1<f64> {
    let mut ssq_freqs = Array1::<f64>::zeros(n_freqs);
    
    // Linear distribution from 0 to fs/2
    for i in 0..n_freqs {
        ssq_freqs[i] = (i as f64) * 0.5 * fs / (n_freqs as f64 - 1.0);
    }
    
    ssq_freqs
}

/// Synchrosqueezed Short-Time Fourier Transform
/// 
/// # Arguments:
/// * `x` - Input signal
/// * `window` - STFT window function
/// * `n_fft` - FFT length
/// * `win_len` - Window length
/// * `hop_len` - STFT stride
/// * `fs` - Sampling frequency
/// * `padtype` - Padding scheme ('reflect' or 'zero')
/// * `squeezing` - Squeezing method ('sum' or 'lebesgue')
/// * `gamma` - CWT phase threshold
/// 
/// # Returns:
/// * `Tx` - Synchrosqueezed STFT
/// * `Sx` - STFT of input
/// * `ssq_freqs` - Frequencies for synchrosqueezed transform
/// * `Sfs` - Frequencies for STFT
#[pyfunction]
#[pyo3(signature = (x, window, n_fft=None, win_len=None, hop_len=1, fs=1.0, padtype="reflect", squeezing="sum", gamma=None))]
pub fn ssq_stft<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    window: PyReadonlyArray1<f64>,
    n_fft: Option<usize>,
    win_len: Option<usize>,
    hop_len: usize,
    fs: f64,
    padtype: &str,
    squeezing: &str,
    gamma: Option<f64>,
) -> PyResult<(PyObject, PyObject, PyObject, PyObject)> {
    // Convert Python arrays to Rust arrays
    let x_array = x.as_array();
    let window_array = window.as_array();
    
    // Process arguments
    let n = x_array.len();
    let n_fft = n_fft.unwrap_or(n.min(512));
    let win_len = win_len.unwrap_or(window_array.len());
    
    // Check window length
    if win_len > n_fft {
        return Err(PyValueError::new_err(format!(
            "Window length {} cannot be greater than n_fft {}", 
            win_len, n_fft
        )));
    }
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Create diff window
        let diff_window = compute_diff_window(&window_array);
        
        // Pad the signal
        let padded_x = match padtype {
            "reflect" => pad_reflect(&x_array, n_fft),
            "zero" => pad_zeros(&x_array, n_fft),
            _ => pad_reflect(&x_array, n_fft), // Default to reflect
        };
        
        // Compute STFT and its derivative
        let Sx = compute_stft(&padded_x, n_fft, hop_len, &window_array);
        let dSx = stft_derivative(&padded_x, n_fft, hop_len, &window_array, &diff_window, fs);
        
        // Compute STFT frequencies
        let n_freqs = n_fft / 2 + 1;
        let Sfs = Array1::<f64>::linspace(0.0, 0.5 * fs, n_freqs);
        
        // Set gamma if not provided
        let gamma = gamma.unwrap_or_else(|| {
            if Sx.is_standard_layout() {
                10.0 * 2.2204460492503131e-16 // 10 * EPS64
            } else {
                10.0 * 1.1920929e-07 // 10 * EPS32
            }
        });
        
        // Compute phase transform
        let w = phase_stft(&Sx, &dSx, &Sfs, gamma);
        
        // Compute frequencies for synchrosqueezed transform
        let ssq_freqs = compute_associated_frequencies(n_freqs, fs, "float64");
        
        // Initialize synchrosqueezed STFT
        let mut Tx = Array2::<Complex64>::zeros((n_freqs, Sx.shape()[1]));
        
        // Frequency bin size for constant in synchrosqueezing
        let dw = ssq_freqs[1] - ssq_freqs[0];
        
        // Synchrosqueezing
        for j in 0..Sx.shape()[1] {
            for i in 0..Sx.shape()[0] {
                if !w[[i, j]].is_infinite() {
                    // Find closest frequency bin
                    let mut k = 0;
                    let mut min_dist = f64::INFINITY;
                    
                    for (idx, &freq) in ssq_freqs.iter().enumerate() {
                        let dist = (w[[i, j]] - freq).abs();
                        if dist < min_dist {
                            min_dist = dist;
                            k = idx;
                        }
                    }
                    
                    // Apply synchrosqueezing
                    let weight = match squeezing {
                        "sum" => Sx[[i, j]],
                        "lebesgue" => Complex64::new(1.0 / (Sx.shape()[0] as f64), 0.0),
                        _ => Sx[[i, j]], // Default to sum
                    };
                    
                    Tx[[k, j]] += weight * dw;
                }
            }
        }
        
        (Tx, Sx, ssq_freqs, Sfs)
    });
    
    // Convert results back to Python objects
    let (Tx, Sx, ssq_freqs, Sfs) = result;
    
    let py_Tx = Tx.into_pyarray(py).to_object(py);
    let py_Sx = Sx.into_pyarray(py).to_object(py);
    let py_ssq_freqs = ssq_freqs.into_pyarray(py).to_object(py);
    let py_Sfs = Sfs.into_pyarray(py).to_object(py);
    
    Ok((py_Tx, py_Sx, py_ssq_freqs, py_Sfs))
}

/// Compute derivative window (for phase transform)
fn compute_diff_window(window: &ArrayView1<f64>) -> Array1<f64> {
    use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};
    
    let n = window.len();
    let mut diff_window = Array1::<f64>::zeros(n);
    
    // Create frequency domain representation for computing derivative
    let mut freqs = Array1::<f64>::zeros(n);
    for i in 0..n/2 + 1 {
        freqs[i] = i as f64;
    }
    for i in n/2 + 1..n {
        freqs[i] = (i as f64) - (n as f64);
    }
    
    // Scale by 2*pi/N for proper frequency domain differentiation
    for i in 0..n {
        freqs[i] *= 2.0 * std::f64::consts::PI / (n as f64);
    }
    
    // Create a planner
    let mut planner = FftPlanner::new();
    let fft_forward = planner.plan_fft_forward(n);
    let fft_inverse = planner.plan_fft_inverse(n);
    
    // Convert window to complex and FFT it
    let mut window_complex: Vec<FFTComplex<f64>> = window
        .iter()
        .map(|&w| FFTComplex::new(w, 0.0))
        .collect();
    
    fft_forward.process(&mut window_complex);
    
    // Multiply by i*omega in frequency domain
    for i in 0..n {
        let re = window_complex[i].re;
        let im = window_complex[i].im;
        window_complex[i] = FFTComplex::new(-im * freqs[i], re * freqs[i]);
    }
    
    // Perform inverse FFT
    fft_inverse.process(&mut window_complex);
    
    // Normalize and convert back to real
    let scale = 1.0 / (n as f64);
    for i in 0..n {
        diff_window[i] = window_complex[i].re * scale;
    }
    
    diff_window
}