// rust/src/spectral/stft_ssq.rs
use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use crate::spectral::stft_utils::{apply_window, pad_reflect, pad_zeros, compute_stft, stft_derivative};

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
) -> Array1<f64> {
    let mut ssq_freqs = Array1::<f64>::zeros(n_freqs);
    
    // Linear distribution from 0 to fs/2
    for i in 0..n_freqs {
        ssq_freqs[i] = (i as f64) * 0.5 * fs / ((n_freqs as f64) - 1.0);
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
/// * `ssq_freqs` - Frequencies for synchrosqueezed transform
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
) -> PyResult<(PyObject, PyObject)> {
    // Convert Python arrays to Rust arrays
    let x_array = x.as_array().to_owned();
    let window_array = window.as_array().to_owned();
    
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

    // Ensure window size matches n_fft by padding if needed
    let window_sized = if window_array.len() < n_fft {
        let pad_left = (n_fft - window_array.len()) / 2;
        let pad_right = n_fft - window_array.len() - pad_left;
        
        let mut padded_window = Array1::<f64>::zeros(n_fft);
        for i in 0..window_array.len() {
            padded_window[i + pad_left] = window_array[i];
        }
        padded_window
    } else if window_array.len() > n_fft {
        let start = (window_array.len() - n_fft) / 2;
        let window_view = window_array.slice(s![start..start+n_fft]);
        window_view.to_owned()
    } else {
        window_array
    };
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Pad the signal using the stft_utils functions
        let padded_x = match padtype {
            "reflect" => pad_reflect(&x_array.view(), n_fft),
            "zero" => pad_zeros(&x_array.view(), n_fft),
            _ => pad_reflect(&x_array.view(), n_fft), // Default to reflect
        };
        
        // Compute the window derivative for STFT derivative calculation
        let diff_window = {
            use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};
            let n = n_fft;
            let mut diff_window = Array1::<f64>::zeros(n);
            
            // Create frequency domain representation
            let mut freqs = Array1::<f64>::zeros(n);
            for i in 0..n/2 + 1 {
                freqs[i] = i as f64;
            }
            for i in n/2 + 1..n {
                freqs[i] = (i as f64) - (n as f64);
            }
            
            // Scale by 2*pi/N
            for i in 0..n {
                freqs[i] *= 2.0 * std::f64::consts::PI / (n as f64);
            }
            
            // FFT the window
            let mut planner = FftPlanner::new();
            let fft_forward = planner.plan_fft_forward(n);
            let fft_inverse = planner.plan_fft_inverse(n);
            
            let mut window_complex: Vec<FFTComplex<f64>> = window_sized
                .iter()
                .map(|&w| FFTComplex::new(w, 0.0))
                .collect();
            
            fft_forward.process(&mut window_complex);
            
            // Multiply by i*omega
            for i in 0..n {
                let re = window_complex[i].re;
                let im = window_complex[i].im;
                window_complex[i] = FFTComplex::new(-im * freqs[i], re * freqs[i]);
            }
            
            // Inverse FFT
            fft_inverse.process(&mut window_complex);
            
            // Normalize
            let scale = 1.0 / (n as f64);
            for i in 0..n {
                diff_window[i] = window_complex[i].re * scale;
            }
            
            diff_window
        };

        // Calculate frames
        let n_samples = padded_x.len();
        let n_frames = ((n_samples - n_fft) / hop_len) + 1;
        let n_freqs = n_fft / 2 + 1;
        
        // Compute STFT
        let mut Sx = Array2::<Complex64>::zeros((n_freqs, n_frames));
        let mut dSx = Array2::<Complex64>::zeros((n_freqs, n_frames));
        
        // Process frames
        let frame_results: Vec<_> = (0..n_frames)
            .into_par_iter()
            .map(|frame| {
                let start = frame * hop_len;
                let frame_slice = padded_x.slice(s![start..start + n_fft]);
                
                // Create a planner for this thread
                let mut planner = rustfft::FftPlanner::new();
                let fft = planner.plan_fft_forward(n_fft);
                
                // Apply window for STFT
                let windowed_frame = apply_window(&frame_slice, &window_sized.view());
                
                // Apply diff_window for STFT derivative
                let diff_windowed_frame = {
                    let mut frame_complex = Array1::<Complex64>::zeros(n_fft);
                    for i in 0..n_fft {
                        frame_complex[i] = Complex64::new(frame_slice[i] * diff_window[i] * fs, 0.0);
                    }
                    frame_complex
                };
                
                // Convert to FFT Complex format for STFT
                let mut stft_input: Vec<rustfft::num_complex::Complex<f64>> = windowed_frame
                    .iter()
                    .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
                    .collect();
                
                // Convert to FFT Complex format for derivative
                let mut dstft_input: Vec<rustfft::num_complex::Complex<f64>> = diff_windowed_frame
                    .iter()
                    .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
                    .collect();
                
                // Perform FFTs
                fft.process(&mut stft_input);
                fft.process(&mut dstft_input);
                
                // Extract the positive frequencies (DC to Nyquist)
                let stft_output: Vec<Complex64> = stft_input
                    .iter()
                    .take(n_freqs)
                    .map(|&c| Complex64::new(c.re, c.im))
                    .collect();
                
                let dstft_output: Vec<Complex64> = dstft_input
                    .iter()
                    .take(n_freqs)
                    .map(|&c| Complex64::new(c.re, c.im))
                    .collect();
                
                (frame, stft_output, dstft_output)
            })
            .collect();
        
        // Combine the results
        for (frame, stft_output, dstft_output) in frame_results {
            for i in 0..stft_output.len() {
                Sx[[i, frame]] = stft_output[i];
                dSx[[i, frame]] = dstft_output[i];
            }
        }
        
        // Compute STFT frequencies
        let Sfs = Array1::<f64>::linspace(0.0, 0.5 * fs, n_freqs);
        
        // Set gamma if not provided
        let gamma = gamma.unwrap_or_else(|| {
            // Use epsilon based on data type
            10.0 * 2.2204460492503131e-16 // 10 * EPS64
        });
        
        // Compute phase transform
        let w = phase_stft(&Sx, &dSx, &Sfs, gamma);
        
        // Compute frequencies for synchrosqueezed transform
        let ssq_freqs = compute_associated_frequencies(n_freqs, fs);
        
        // Initialize synchrosqueezed STFT
        let mut Tx = Array2::<Complex64>::zeros((n_freqs, n_frames));
        
        // Frequency bin size for constant in synchrosqueezing
        let dw = ssq_freqs[1] - ssq_freqs[0];
        
        // Synchrosqueezing
        for j in 0..n_frames {
            for i in 0..n_freqs {
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
                        "lebesgue" => Complex64::new(1.0 / (n_freqs as f64), 0.0),
                        _ => Sx[[i, j]], // Default to sum
                    };
                    
                    Tx[[k, j]] += weight * dw;
                }
            }
        }
        
        (Tx, ssq_freqs)
    });
    
    // Convert results back to Python objects
    let (Tx, ssq_freqs) = result;
    
    let py_Tx = Tx.into_pyarray(py).to_object(py);
    let py_ssq_freqs = ssq_freqs.into_pyarray(py).to_object(py);
    
    Ok((py_Tx, py_ssq_freqs))
}