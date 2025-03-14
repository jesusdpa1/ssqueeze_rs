// rust/src/wavelets/base.rs
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use numpy::IntoPyArray;
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use pyo3::types::PyDict;
use std::f64::consts::PI;

/// Generate frequency-domain grid for wavelet computation
/// 
/// # Arguments
/// * `scale` - Scale parameter
/// * `n` - Length of the wavelet
/// 
/// # Returns
/// * `xi` - Frequency-domain grid
pub fn xifn(scale: f64, n: usize) -> Array1<f64> {
    let mut xi = Array1::zeros(n);
    let h = scale * (2.0 * PI) / (n as f64);
    
    // First half: [0, 1, 2, ..., n/2]
    for i in 0..n/2 + 1 {
        xi[i] = i as f64 * h;
    }
    
    // Second half: [-(n/2-1), -(n/2-2), ..., -1]
    for i in n/2 + 1..n {
        xi[i] = (i as isize - n as isize) as f64 * h;
    }
    
    xi
}

/// Base trait for all wavelets
pub trait WaveletBase {
    /// Compute wavelet in frequency domain
    fn psih(&self, xi: &ArrayView1<f64>) -> Array1<Complex64>;
    
    /// Get the center frequency
    fn center_frequency(&self) -> f64;
    
    /// Get the name of the wavelet
    fn name(&self) -> String;
}

/// Wavelet transform options
pub struct WaveletTransformOptions {
    pub l1_norm: bool,
    pub derivative: bool,
    pub vectorized: bool,
    pub cache_wavelet: bool,
}

impl Default for WaveletTransformOptions {
    fn default() -> Self {
        WaveletTransformOptions {
            l1_norm: true,
            derivative: false,
            vectorized: true,
            cache_wavelet: true,
        }
    }
}

/// Helper function to convert Python dtypes to Rust types
pub fn get_dtype_from_str(dtype_str: &str) -> Result<&'static str, PyErr> {
    match dtype_str {
        "float32" => Ok("f32"),
        "float64" => Ok("f64"),
        "complex64" => Ok("c64"),
        "complex128" => Ok("c128"),
        _ => Err(PyValueError::new_err(format!("Unsupported dtype: {}", dtype_str))),
    }
}

/// Process wavelet parameters from Python
#[pyfunction]
pub fn process_wavelet_params<'py>(
    py: Python<'py>,
    wavelet_type: &str,
    params: &Bound<'py, PyAny>,
) -> PyResult<PyObject> {
    // Extract parameters from Python dict
    let params_dict = params.downcast::<PyDict>()?;
    
    // Return processed parameters as Python dict
    let result = PyDict::new(py);
    
    // Common parameters for all wavelets
    if let Some(dtype) = params_dict.get_item("dtype")? {
        result.set_item("dtype", dtype)?;
    } else {
        // Default dtype
        result.set_item("dtype", "float64")?;
    }
    
    // Process specific wavelet parameters
    match wavelet_type {
        "morlet" => {
            // Default mu value
            let mu = if let Some(mu_val) = params_dict.get_item("mu")? {
                mu_val.extract::<f64>()?
            } else {
                6.0
            };
            result.set_item("mu", mu)?;
        },
        "gmw" => {
            // Default GMW parameters
            let gamma = if let Some(gamma_val) = params_dict.get_item("gamma")? {
                gamma_val.extract::<f64>()?
            } else {
                3.0
            };
            let beta = if let Some(beta_val) = params_dict.get_item("beta")? {
                beta_val.extract::<f64>()?
            } else {
                60.0
            };
            let norm = if let Some(norm_val) = params_dict.get_item("norm")? {
                norm_val.extract::<String>()?
            } else {
                "bandpass".to_string()
            };
            let order = if let Some(order_val) = params_dict.get_item("order")? {
                order_val.extract::<i32>()?
            } else {
                0
            };
            
            result.set_item("gamma", gamma)?;
            result.set_item("beta", beta)?;
            result.set_item("norm", norm)?;
            result.set_item("order", order)?;
        },
        "bump" => {
            // Default bump parameters
            let mu = if let Some(mu_val) = params_dict.get_item("mu")? {
                mu_val.extract::<f64>()?
            } else {
                5.0
            };
            let sigma = if let Some(sigma_val) = params_dict.get_item("sigma")? {
                sigma_val.extract::<f64>()?
            } else {
                1.0
            };
            
            result.set_item("mu", mu)?;
            result.set_item("sigma", sigma)?;
        },
        _ => {
            return Err(PyValueError::new_err(format!("Unsupported wavelet type: {}", wavelet_type)));
        }
    }
    
    Ok(result.into_py(py))
}

/// Helper function to convert between various norm types
pub fn normalize_wavelet(
    psih: &mut Array1<Complex64>, 
    from_norm: &str, 
    to_norm: &str, 
    scale: f64
) {
    if from_norm == to_norm {
        return;
    }
    
    match (from_norm, to_norm) {
        ("energy", "bandpass") => {
            // Find the peak value
            let mut max_val = 0.0;
            for &val in psih.iter() {
                let abs_val = val.norm();
                if abs_val > max_val {
                    max_val = abs_val;
                }
            }
            
            // Scale to have peak value of 2.0
            if max_val > 0.0 {
                let scale_factor = 2.0 / max_val;
                for val in psih.iter_mut() {
                    *val *= scale_factor;
                }
            }
        },
        ("bandpass", "energy") => {
            // L1 to L2 normalization
            if scale > 0.0 {
                let scale_factor = scale.sqrt();
                for val in psih.iter_mut() {
                    *val /= scale_factor;
                }
            }
        },
        _ => {
            // Other combinations not supported yet
        }
    }
}