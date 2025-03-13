// rust/src/utils/mod.rs
use ndarray::{Array1, Array2, ArrayView1, s};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rustfft::FftPlanner;
use num_complex::Complex64;

/// Generate wavelet scales
/// 
/// # Arguments
/// * `min_scale` - Minimum scale
/// * `max_scale` - Maximum scale
/// * `n_scales` - Number of scales
/// * `scale_type` - Type of scaling ('log' or 'linear')
/// 
/// # Returns
/// * `scales` - Array of scales
#[pyfunction]
pub fn generate_scales<'py>(
    py: Python<'py>,
    min_scale: f64,
    max_scale: f64,
    n_scales: usize,
    scale_type: &str,
) -> PyResult<PyObject> {
    let scales = match scale_type {
        "log" => {
            let log_min = min_scale.ln();
            let log_max = max_scale.ln();
            let step = (log_max - log_min) / ((n_scales - 1) as f64);
            
            let mut scales = Array1::<f64>::zeros(n_scales);
            for i in 0..n_scales {
                scales[i] = (log_min + i as f64 * step).exp();
            }
            scales
        },
        "linear" => {
            let step = (max_scale - min_scale) / ((n_scales - 1) as f64);
            
            let mut scales = Array1::<f64>::zeros(n_scales);
            for i in 0..n_scales {
                scales[i] = min_scale + i as f64 * step;
            }
            scales
        },
        _ => return Err(PyValueError::new_err(format!("Unknown scale type: {}", scale_type))),
    };
    
    Ok(scales.into_pyarray(py).to_object(py))
}

/// Generate frequency domain wavelet
/// 
/// # Arguments
/// * `n` - Length of wavelet
/// * `scale` - Scale parameter
/// * `wavelet_type` - Type of wavelet ('morlet', 'bump', etc.)
/// * `param` - Additional parameter for wavelet customization
/// 
/// # Returns
/// * `wavelet` - Complex wavelet in frequency domain
#[pyfunction]
pub fn frequency_wavelet<'py>(
    py: Python<'py>,
    n: usize,
    scale: f64,
    wavelet_type: &str,
    param: Option<f64>,
) -> PyResult<PyObject> {
    let param = param.unwrap_or(5.0); // Default parameter value
    
    let mut wavelet = Array1::<Complex64>::zeros(n);
    
    // Generate frequency domain
    let mut omega = Array1::<f64>::zeros(n);
    for i in 0..n/2+1 {
        omega[i] = 2.0 * std::f64::consts::PI * (i as f64) / (n as f64);
    }
    for i in n/2+1..n {
        omega[i] = 2.0 * std::f64::consts::PI * ((i as f64) - (n as f64)) / (n as f64);
    }
    
    // Scale the frequencies
    for i in 0..n {
        omega[i] *= scale;
    }
    
    // Generate wavelet
    match wavelet_type {
        "morlet" => {
            // Morlet wavelet (complex Gaussian modulated by exponential)
            let mu = param;
            for i in 0..n {
                let w = omega[i];
                // Only positive frequencies for analytic wavelet
                if w >= 0.0 {
                    let term1 = (-0.5 * (w - mu).powi(2)).exp();
                    let term2 = (-0.5 * mu.powi(2)).exp(); // Admissibility correction
                    wavelet[i] = Complex64::new(term1 - term2, 0.0);
                }
            }
        },
        "bump" => {
            // Bump wavelet
            let sigma = param;
            for i in 0..n {
                let w = omega[i];
                // Only positive frequencies for analytic wavelet
                if w > 0.0 && w < 1.0 {
                    let term = (-1.0 / (1.0 - (2.0*w - 1.0).powi(2))).exp() / sigma;
                    wavelet[i] = Complex64::new(term, 0.0);
                }
            }
        },
        _ => return Err(PyValueError::new_err(format!("Unknown wavelet type: {}", wavelet_type))),
    };
    
    Ok(wavelet.into_pyarray(py).to_object(py))
}

/// Perform padding on signal
/// 
/// # Arguments
/// * `x` - Input signal
/// * `pad_len` - Length after padding
/// * `pad_type` - Type of padding ('reflect', 'zero', etc.)
/// 
/// # Returns
/// * `x_padded` - Padded signal
#[pyfunction]
pub fn pad_signal<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    pad_len: usize,
    pad_type: &str,
) -> PyResult<PyObject> {
    let x_array = x.as_array();
    let n = x_array.len();
    
    if pad_len <= n {
        return Err(PyValueError::new_err("pad_len must be greater than length of x"));
    }
    
    let pad_left = (pad_len - n) / 2;
    let pad_right = pad_len - n - pad_left;
    
    let mut x_padded = Array1::<f64>::zeros(pad_len);
    
    // Copy original signal to middle
    for i in 0..n {
        x_padded[i + pad_left] = x_array[i];
    }
    
    // Apply padding
    match pad_type {
        "reflect" => {
            // Left padding
            for i in 0..pad_left {
                let idx = pad_left - i;
                if idx < n {
                    x_padded[i] = x_array[idx];
                }
            }
            
            // Right padding
            for i in 0..pad_right {
                let idx = n - 2 - i;
                if idx >= 0 && idx < n {
                    x_padded[n + pad_left + i] = x_array[idx];
                }
            }
        },
        "zero" => {
            // Already filled with zeros
        },
        _ => return Err(PyValueError::new_err(format!("Unknown padding type: {}", pad_type))),
    }
    
    Ok(x_padded.into_pyarray(py).to_object(py))
}

/// Compute next power of 2
#[pyfunction]
pub fn next_power_of_2(n: usize) -> usize {
    1 << (n as f64).log2().ceil() as usize
}

/// Power of 2 up
/// 
/// Calculates next power of 2, and left/right padding to center
/// the original `n` locations.
#[pyfunction]
pub fn p2up(n: usize) -> PyResult<(usize, usize, usize)> {
    let up = 1 << ((n as f64).log2().ceil() as usize);
    let n2 = (up - n) / 2;
    let n1 = up - n - n2;
    
    Ok((up, n1, n2))
}