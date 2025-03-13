use ndarray::{Array2, Axis};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute classic Teager-Kaiser Energy Operator (TKEO) for 1D signals
///
/// TKEO = x[n]² - x[n-1] * x[n+1]
#[pyfunction]
pub fn compute_tkeo_classic(
    py: Python<'_>,
    signal: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    // Convert input to rust ndarray
    let signal_array = signal.as_array();
    let signal_len = signal_array.len();
    
    // Need at least 3 points for TKEO
    if signal_len < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal must have at least 3 points for TKEO calculation",
        ));
    }
    
    // Allocate output array (2 samples shorter than input)
    let mut output = Array2::zeros((signal_len - 2, 1));
    
    // Compute TKEO
    for i in 1..(signal_len - 1) {
        let x_curr = signal_array[i];
        let x_prev = signal_array[i - 1];
        let x_next = signal_array[i + 1];
        
        output[[i - 1, 0]] = x_curr * x_curr - x_prev * x_next;
    }
    
    // Return as 1D array
    let output_1d = output.index_axis_move(Axis(1), 0);
    Ok(output_1d.into_pyarray(py).to_owned().into())
}

/// Compute modified Teager-Kaiser Energy Operator (TKEO) for 1D signals
///
/// Modified TKEO = x[n+1]*x[n-2] - x[n]*x[n-3]
#[pyfunction]
pub fn compute_tkeo_modified(
    py: Python<'_>,
    signal: PyReadonlyArray1<f32>,
) -> PyResult<Py<PyArray1<f32>>> {
    // Convert input to rust ndarray
    let signal_array = signal.as_array();
    let signal_len = signal_array.len();
    
    // Need at least 5 points for modified TKEO
    if signal_len < 5 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal must have at least 5 points for modified TKEO calculation",
        ));
    }
    
    // Allocate output array (3 samples shorter than input)
    let mut output = Array2::zeros((signal_len - 3, 1));
    
    // Constants from Deburchgrave et al., 2008
    let l = 1; // Parameter l
    let p = 2; // Parameter p
    let q = 0; // Parameter q
    let s = 3; // Parameter s
    
    // Compute modified TKEO
    for i in s..(signal_len) {
        let pos1 = i - l;
        let pos2 = i - p;
        let pos3 = i - q;
        let pos4 = i - s;
        
        // Since positions are usize (unsigned), they can't be negative
        // Just do the computation - we've already checked signal_len >= 5
        output[[pos4, 0]] = signal_array[pos1] * signal_array[pos2] - signal_array[pos3] * signal_array[pos4];
    }
    
    // Return as 1D array
    let output_1d = output.index_axis_move(Axis(1), 0);
    Ok(output_1d.into_pyarray(py).to_owned().into())
}

/// Apply classic TKEO to 2D multichannel data
#[pyfunction]
pub fn compute_tkeo(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Convert input to rust ndarray
    let data_array = data.as_array();
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Need at least 3 points for TKEO
    if n_samples < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal must have at least 3 points for TKEO calculation",
        ));
    }
    
    // Allocate output array (2 samples shorter than input)
    let mut output = Array2::zeros((n_samples - 2, n_channels));
    
    // Process each channel
    for c in 0..n_channels {
        // Compute TKEO
        for i in 1..(n_samples - 1) {
            let x_curr = data_array[[i, c]];
            let x_prev = data_array[[i - 1, c]];
            let x_next = data_array[[i + 1, c]];
            
            output[[i - 1, c]] = x_curr * x_curr - x_prev * x_next;
        }
    }
    
    Ok(output.into_pyarray(py).to_owned().into())
}

/// Apply classic TKEO to 2D multichannel data with parallel processing
#[pyfunction]
pub fn compute_tkeo_parallel(
    py: Python<'_>,
    data: PyReadonlyArray2<f32>,
) -> PyResult<Py<PyArray2<f32>>> {
    // Extract data to a Rust array before releasing the GIL
    let data_array = data.as_array().to_owned();
    let (n_samples, n_channels) = (data_array.shape()[0], data_array.shape()[1]);
    
    // Need at least 3 points for TKEO
    if n_samples < 3 {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "Signal must have at least 3 points for TKEO calculation",
        ));
    }
    
    // Allow Python threads to run during computation
    let output = Python::allow_threads(py, || {
        // Create the output array
        let mut output = Array2::zeros((n_samples - 2, n_channels));
        
        // Process each channel in parallel using rayon
        use rayon::iter::ParallelIterator;
        let channel_results: Vec<_> = (0..n_channels)
            .into_par_iter()
            .map(|c| {
                // Create a separate array for each channel
                let mut channel_output = vec![0.0f32; n_samples - 2];
                
                // Compute TKEO for this channel
                for i in 1..(n_samples - 1) {
                    let x_curr = data_array[[i, c]];
                    let x_prev = data_array[[i - 1, c]];
                    let x_next = data_array[[i + 1, c]];
                    
                    channel_output[i - 1] = x_curr * x_curr - x_prev * x_next;
                }
                
                (c, channel_output)
            })
            .collect();
        
        // Combine results from all channels
        for (c, channel_data) in channel_results {
            for i in 0..(n_samples - 2) {
                output[[i, c]] = channel_data[i];
            }
        }
        
        output
    });
    
    // Convert the result back to Python
    Ok(output.into_pyarray(py).to_owned().into())
}