// rust/src/spectral/stft.rs
use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;

use crate::spectral::stft_utils::{apply_window, pad_reflect, pad_zeros, compute_diff_window};

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
    // Convert Python arrays to Rust arrays and make owned copies
    let x_array = x.as_array().to_owned();
    let window_array = window.as_array().to_owned();
    
    // Pad the signal if needed
    let padded_x = match padtype {
        "reflect" => pad_reflect(&x_array.view(), n_fft),
        "zero" => pad_zeros(&x_array.view(), n_fft),
        _ => pad_reflect(&x_array.view(), n_fft), // Default to reflect
    };
    
    // Calculate the number of frames
    let n_samples = padded_x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    // Allow Python threads to run during computation
    let (stft_result, freqs) = Python::allow_threads(py, || {
        // Initialize output arrays
        let mut stft_result = Array2::<Complex64>::zeros((n_freqs, n_frames));
        let freqs = Array1::<f64>::linspace(0.0, 0.5, n_freqs);
        
        // Create FFT planner for reuse
        let mut planner = rustfft::FftPlanner::new();
        let fft_plan = planner.plan_fft_forward(n_fft);
        
        // Process frames in parallel
        let frame_results: Vec<_> = (0..n_frames)
            .into_par_iter()
            .map(|frame| {
                // Extract the frame
                let start = frame * hop_length;
                let end = start + n_fft;
                
                // Create a view of this segment
                let frame_view = padded_x.slice(s![start..end]);
                
                // Apply window
                let windowed_frame = apply_window(&frame_view, &window_array.view());
                
                // Convert to FFT Complex format
                let mut fft_input: Vec<rustfft::num_complex::Complex<f64>> = windowed_frame
                    .iter()
                    .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
                    .collect();
                
                // Perform FFT
                fft_plan.process(&mut fft_input);
                
                // Extract the positive frequencies (DC to Nyquist)
                let frame_output: Vec<Complex64> = fft_input
                    .iter()
                    .take(n_freqs)
                    .map(|&c| Complex64::new(c.re, c.im))
                    .collect();
                
                (frame, frame_output)
            })
            .collect();
        
        // Combine the results
        for (frame, frame_output) in frame_results {
            for (i, &value) in frame_output.iter().enumerate() {
                stft_result[[i, frame]] = value;
            }
        }
        
        (stft_result, freqs)
    });

    // Convert back to Python objects
    let py_stft = stft_result.into_pyarray(py).to_object(py);
    let py_freqs = freqs.into_pyarray(py).to_object(py);
    
    Ok((py_stft, py_freqs))
}

use rayon::prelude::*;