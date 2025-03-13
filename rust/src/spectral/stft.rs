// rust/src/spectral/stft.rs
use ndarray::{Array1, Array2, ArrayView1, s, Axis};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};
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
        let mut planner = FftPlanner::new();
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
                let mut windowed_frame = apply_window(&frame_view, &window_array.view());
                
                // Convert to FFT Complex format
                let mut fft_input: Vec<FFTComplex<f64>> = windowed_frame
                    .iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
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
    padded_x: &Array1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: &ArrayView1<f64>,
    diff_window: &ArrayView1<f64>,
    fs: f64,
) -> Array2<Complex64> {
    // Similar to stft but using the derivative of the window
    let n_samples = padded_x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    let mut d_stft = Array2::<Complex64>::zeros((n_freqs, n_frames));
    
    // Process frames in parallel
    let frame_results: Vec<_> = (0..n_frames)
        .into_par_iter()
        .map(|frame| {
            let start = frame * hop_length;
            let frame_slice = padded_x.slice(s![start..start + n_fft]);
            
            // Create a planner for this thread
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n_fft);
            
            // Apply derivative window
            let mut windowed_frame = apply_window(&frame_slice, &diff_window);
            
            // Scale by sampling frequency
            for i in 0..windowed_frame.len() {
                windowed_frame[i] *= fs;
            }
            
            // Convert to FFT Complex format
            let mut fft_input: Vec<FFTComplex<f64>> = windowed_frame
                .iter()
                .map(|&c| FFTComplex::new(c.re, c.im))
                .collect();
            
            // Perform FFT
            fft.process(&mut fft_input);
            
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
            d_stft[[i, frame]] = value;
        }
    }
    
    d_stft
}

// Helper function to compute full STFT (internal use)
fn compute_stft(
    padded_x: &Array1<f64>,
    n_fft: usize,
    hop_length: usize,
    window: &ArrayView1<f64>,
) -> Array2<Complex64> {
    let n_samples = padded_x.len();
    let n_frames = ((n_samples - n_fft) / hop_length) + 1;
    let n_freqs = n_fft / 2 + 1;
    
    let mut stft_result = Array2::<Complex64>::zeros((n_freqs, n_frames));
    
    // Process frames in parallel
    let frame_results: Vec<_> = (0..n_frames)
        .into_par_iter()
        .map(|frame| {
            let start = frame * hop_length;
            let frame_slice = padded_x.slice(s![start..start + n_fft]);
            
            // Create a planner for this thread
            let mut planner = FftPlanner::new();
            let fft = planner.plan_fft_forward(n_fft);
            
            // Apply window
            let mut windowed_frame = apply_window(&frame_slice, window);
            
            // Convert to FFT Complex format
            let mut fft_input: Vec<FFTComplex<f64>> = windowed_frame
                .iter()
                .map(|&c| FFTComplex::new(c.re, c.im))
                .collect();
            
            // Perform FFT
            fft.process(&mut fft_input);
            
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
    
    stft_result
}

// Compute derivative window (for phase transform)
fn compute_diff_window(window: &ArrayView1<f64>) -> Array1<f64> {
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
