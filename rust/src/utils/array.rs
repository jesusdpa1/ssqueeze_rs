// rust/src/utils/array.rs
use ndarray::{Array1, ArrayView1, s};
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

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

/// Pad signal with various padding modes
#[pyfunction]
pub fn pad_signal<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    pad_len: usize,
    pad_type: &str,
) -> PyResult<PyObject> {
    let x_array = x.as_array();
    
    if pad_len <= x_array.len() {
        return Err(PyValueError::new_err("pad_len must be greater than length of x"));
    }
    
    let padded = match pad_type {
        "reflect" => pad_reflect(x_array, pad_len),
        "zero" => pad_zero(x_array, pad_len),
        _ => {
            return Err(PyValueError::new_err(format!("Unknown padding type: {}", pad_type)));
        }
    };
    
    Ok(padded.into_pyarray(py).to_object(py))
}

/// Pad signal with reflection at boundaries
pub fn pad_reflect(x: ArrayView1<f64>, pad_len: usize) -> Array1<f64> {
    let n = x.len();
    let pad_size = pad_len - n;
    let pad_left = pad_size / 2;
    let pad_right = pad_size - pad_left;
    
    let mut padded = Array1::<f64>::zeros(pad_len);
    
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
pub fn pad_zero(x: ArrayView1<f64>, pad_len: usize) -> Array1<f64> {
    let n = x.len();
    let pad_size = pad_len - n;
    let pad_left = pad_size / 2;
    
    let mut padded = Array1::<f64>::zeros(pad_len);
    
    // Copy original signal to the middle
    for i in 0..n {
        padded[i + pad_left] = x[i];
    }
    
    padded
}