// rust/src/spectral/cwt.rs
use ndarray::{Array1, Array2, ArrayView1, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};
use crate::utils::array::{pad_reflect, pad_zero, next_power_of_2};
use crate::wavelets::base::xifn;
use std::sync::Arc;

/// Compute the continuous wavelet transform
/// 
/// # Arguments
/// * `x` - Input signal (1D array)
/// * `wavelet` - Wavelet function or predefined wavelet name
/// * `scales` - Scales at which to compute the CWT
/// * `fs` - Sampling frequency
/// * `t` - Time vector
/// * `nv` - Number of voices per octave
/// * `l1_norm` - Whether to use L1 normalization
/// * `derivative` - Whether to also compute the derivative of the CWT
/// * `padtype` - Type of padding to use
/// * `rpadded` - Whether to return padded CWT
/// 
/// # Returns
/// * `Wx` - CWT of x
/// * `scales` - Scales at which CWT was computed
/// * `dWx` - Time-derivative of CWT of x, if requested
#[pyfunction]
#[pyo3(signature = (
    x, 
    wavelet="gmw", 
    scales=None, 
    fs=None, 
    t=None, 
    nv=32, 
    l1_norm=true, 
    derivative=false, 
    padtype="reflect", 
    rpadded=false, 
    vectorized=true,
    patience=0
))]
pub fn cwt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    wavelet: &str,
    scales: Option<PyReadonlyArray1<f64>>,
    fs: Option<f64>,
    t: Option<PyReadonlyArray1<f64>>,
    nv: usize,
    l1_norm: bool,
    derivative: bool,
    padtype: &str,
    rpadded: bool,
    vectorized: bool,
    patience: i32,
) -> PyResult<(PyObject, PyObject, Option<PyObject>)> {
    // Convert Python arrays to Rust arrays
    let x_array = x.as_array().to_owned();
    let N = x_array.len();
    
    // Process sampling frequency and time vector
    let dt = if let Some(t_array) = t {
        let t_values = t_array.as_array();
        if t_values.len() < 2 {
            return Err(PyValueError::new_err("Time vector must have at least 2 elements"));
        }
        t_values[1] - t_values[0]
    } else if let Some(fs_value) = fs {
        1.0 / fs_value
    } else {
        1.0 // Default sampling period
    };
    
    // Process scales
    let scales_array = match scales {
        Some(s) => s.as_array().to_owned(),
        None => generate_log_scales(N, nv, wavelet)?
    };
    
    // Allow Python threads to run during computation
    let result = Python::allow_threads(py, || {
        // Power of 2 padding
        let pad_len = next_power_of_2(N + N/2); // Add extra padding for better edge handling
        let padded_x = match padtype {
            "reflect" => pad_reflect(x_array.view(), pad_len),
            "zero" => pad_zero(x_array.view(), pad_len),
            _ => pad_reflect(x_array.view(), pad_len) // Default to reflect
        };
        
        // Convert to frequency domain
        let x_freq = compute_fft(&padded_x);
        
        // Calculate padding indices for unpadding later
        let n1 = (pad_len - N) / 2;
        
        // Compute CWT
        let (Wx, dWx) = if vectorized {
            compute_cwt_vectorized(&x_freq, &scales_array, N, pad_len, n1, dt, l1_norm, derivative, wavelet)
        } else {
            compute_cwt_loop(&x_freq, &scales_array, N, pad_len, n1, dt, l1_norm, derivative, wavelet)
        };
        
        // Return full padded result or unpadded
        let (Wx_result, dWx_result) = if rpadded {
            (Wx, dWx)
        } else {
            // Extract only the portion corresponding to the original signal
            let Wx_unpadded = if Wx.is_empty() {
                Array2::<Complex64>::zeros((0, 0))
            } else {
                Wx.slice(s![.., n1..(n1 + N)]).to_owned()
            };
            
            let dWx_unpadded = if let Some(d) = dWx {
                if d.is_empty() {
                    None
                } else {
                    Some(d.slice(s![.., n1..(n1 + N)]).to_owned())
                }
            } else {
                None
            };
            
            (Wx_unpadded, dWx_unpadded)
        };
        
        (Wx_result, dWx_result)
    });
    
    // Convert results to Python objects
    let (Wx, dWx) = result;
    let py_scales = scales_array.into_pyarray(py).to_object(py);
    let py_Wx = Wx.into_pyarray(py).to_object(py);
    let py_dWx = match dWx {
        Some(d) => Some(d.into_pyarray(py).to_object(py)),
        None => None
    };
    
    Ok((py_Wx, py_scales, py_dWx))
}

// Helper function to compute FFT
fn compute_fft(x: &Array1<f64>) -> Array1<Complex64> {
    let pad_len = x.len();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(pad_len);
    
    let mut x_fft: Vec<FFTComplex<f64>> = x.iter()
        .map(|&x| FFTComplex::new(x, 0.0))
        .collect();
    
    fft.process(&mut x_fft);
    
    // Convert to ndarray for easier manipulation
    Array1::<Complex64>::from_iter(
        x_fft.iter().map(|&c| Complex64::new(c.re, c.im))
    )
}

// No longer needed since we're using a different approach
// for thread safety with rustfft


// Optimized vectorized implementation
fn compute_cwt_vectorized(
    x_freq: &Array1<Complex64>,
    scales: &Array1<f64>,
    N: usize,
    pad_len: usize,
    n1: usize,
    dt: f64,
    l1_norm: bool,
    derivative: bool,
    wavelet: &str
) -> (Array2<Complex64>, Option<Array2<Complex64>>) {
    let num_scales = scales.len();
    let mut Wx = Array2::<Complex64>::zeros((num_scales, pad_len));
    let mut dWx = if derivative {
        Some(Array2::<Complex64>::zeros((num_scales, pad_len)))
    } else {
        None
    };
    
    // Compute frequency domain points once
    let xi = xifn(1.0, pad_len);
    
    // Pre-compute wavelets in parallel
    let wavelets: Vec<_> = scales.iter().enumerate()
        .map(|(i, &scale)| (i, scale))
        .collect::<Vec<_>>()
        .par_iter()
        .map(|&(i, scale)| {
            (i, generate_wavelet_fourier(&xi, scale, wavelet))
        })
        .collect();
    
    // Pre-compute derivatives if needed
    let derivatives = if derivative {
        let derivs: Vec<_> = wavelets.par_iter()
            .map(|&(i, ref psih)| {
                let mut dpsih = psih.clone();
                for (j, &w) in xi.iter().enumerate() {
                    dpsih[j] *= Complex64::new(0.0, w / dt);
                }
                (i, dpsih)
            })
            .collect();
        Some(derivs)
    } else {
        None
    };
    
    // Process scales in parallel but with thread-local FFT planners
    // Determine chunk size based on number of scales
    let chunk_size = if num_scales <= 8 {
        1 // For small scale counts, process individually
    } else if num_scales <= 32 {
        4 // Medium sized batches for moderate scale counts
    } else {
        8 // Larger batches for many scales
    };
    
    // Create thread-local results
    let results: Vec<_> = wavelets.par_iter()
        .map(|&(scale_idx, ref psih)| {
            // Create thread-local FFT planner
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(pad_len);
            
            // Create a buffer for the frequency domain product
            let mut Wx_freq = vec![Complex64::new(0.0, 0.0); pad_len];
            
            // Compute convolution in frequency domain
            for i in 0..pad_len {
                Wx_freq[i] = x_freq[i] * psih[i];
            }
            
            // Prepare for inverse FFT
            let mut Wx_fft: Vec<FFTComplex<f64>> = Wx_freq.iter()
                .map(|&c| FFTComplex::new(c.re, c.im))
                .collect();
            
            // Compute inverse FFT
            ifft.process(&mut Wx_fft);
            
            // Normalize
            let norm_factor = 1.0 / (pad_len as f64);
            let scale = scales[scale_idx];
            let scale_factor = if !l1_norm { scale.sqrt() } else { 1.0 };
            
            let Wx_time: Vec<Complex64> = Wx_fft.iter()
                .map(|&c| {
                    Complex64::new(
                        c.re * norm_factor * scale_factor,
                        c.im * norm_factor * scale_factor
                    )
                })
                .collect();
            
            // Process derivative if requested
            let dWx_time = if derivative {
                if let Some(ref derivatives_vec) = derivatives {
                    // Find the derivative for this scale
                    if let Some((_, ref dpsih)) = derivatives_vec.iter()
                        .find(|&&(idx, _)| idx == scale_idx) 
                    {
                        // Create a buffer for the frequency domain product
                        let mut dWx_freq = vec![Complex64::new(0.0, 0.0); pad_len];
                        
                        // Compute convolution in frequency domain
                        for i in 0..pad_len {
                            dWx_freq[i] = x_freq[i] * dpsih[i];
                        }
                        
                        // Prepare for inverse FFT
                        let mut dWx_fft: Vec<FFTComplex<f64>> = dWx_freq.iter()
                            .map(|&c| FFTComplex::new(c.re, c.im))
                            .collect();
                        
                        // Compute inverse FFT
                        ifft.process(&mut dWx_fft);
                        
                        // Normalize
                        let dWx_time: Vec<Complex64> = dWx_fft.iter()
                            .map(|&c| {
                                Complex64::new(
                                    c.re * norm_factor * scale_factor,
                                    c.im * norm_factor * scale_factor
                                )
                            })
                            .collect();
                        
                        Some(dWx_time)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };
            
            (scale_idx, Wx_time, dWx_time)
        })
        .collect();
    
    // Combine results
    for (scale_idx, Wx_time, dWx_time) in results {
        for j in 0..pad_len {
            Wx[[scale_idx, j]] = Wx_time[j];
        }
        
        if let (Some(ref mut dWx_arr), Some(ref dWx_t)) = (&mut dWx, &dWx_time) {
            for j in 0..pad_len {
                dWx_arr[[scale_idx, j]] = dWx_t[j];
            }
        }
    }
    
    (Wx, dWx)
}

// Optimized loop implementation
fn compute_cwt_loop(
    x_freq: &Array1<Complex64>,
    scales: &Array1<f64>,
    N: usize,
    pad_len: usize,
    n1: usize,
    dt: f64,
    l1_norm: bool,
    derivative: bool,
    wavelet: &str
) -> (Array2<Complex64>, Option<Array2<Complex64>>) {
    let num_scales = scales.len();
    let mut Wx = Array2::<Complex64>::zeros((num_scales, pad_len));
    let mut dWx = if derivative {
        Some(Array2::<Complex64>::zeros((num_scales, pad_len)))
    } else {
        None
    };
    
    // Generate frequency domain points
    let xi = xifn(1.0, pad_len);
    
    // Create a reusable planner and transformation plan
    let mut planner = FftPlanner::new();
    let ifft = planner.plan_fft_inverse(pad_len);
    
    // Pre-allocate buffers for FFT operations
    let mut Wx_freq = Vec::with_capacity(pad_len);
    let mut Wx_fft = Vec::with_capacity(pad_len);
    
    let mut dWx_freq = if derivative {
        Some(Vec::with_capacity(pad_len))
    } else {
        None
    };
    let mut dWx_fft = if derivative {
        Some(Vec::with_capacity(pad_len))
    } else {
        None
    };
    
    // Initialize vectors to full size
    Wx_freq.resize(pad_len, Complex64::new(0.0, 0.0));
    Wx_fft.resize(pad_len, FFTComplex::new(0.0, 0.0));
    
    if let Some(ref mut v) = dWx_freq {
        v.resize(pad_len, Complex64::new(0.0, 0.0));
    }
    if let Some(ref mut v) = dWx_fft {
        v.resize(pad_len, FFTComplex::new(0.0, 0.0));
    }
    
    // Process each scale
    for (scale_idx, &scale) in scales.iter().enumerate() {
        // Generate wavelet at this scale
        let psih = generate_wavelet_fourier(&xi, scale, wavelet);
        
        // Compute convolution in frequency domain
        for i in 0..pad_len {
            Wx_freq[i] = x_freq[i] * psih[i];
        }
        
        // Convert to FFT format
        for i in 0..pad_len {
            Wx_fft[i] = FFTComplex::new(Wx_freq[i].re, Wx_freq[i].im);
        }
        
        // Transform back to time domain
        ifft.process(&mut Wx_fft);
        
        // Normalize and store
        let norm_factor = 1.0 / (pad_len as f64);
        for j in 0..pad_len {
            let val = Complex64::new(
                Wx_fft[j].re * norm_factor,
                Wx_fft[j].im * norm_factor
            );
            
            // Apply L1/L2 normalization
            Wx[[scale_idx, j]] = if l1_norm {
                val
            } else {
                val * scale.sqrt() // L2 norm
            };
        }
        
        // Compute derivative if requested
        if derivative {
            // For frequency domain differentiation
            let mut dpsih = Array1::<Complex64>::zeros(pad_len);
            for (i, &w) in xi.iter().enumerate() {
                dpsih[i] = psih[i] * Complex64::new(0.0, w / dt);
            }
            
            // Compute convolution
            if let (Some(ref mut df), Some(ref mut dff)) = (&mut dWx_freq, &mut dWx_fft) {
                for i in 0..pad_len {
                    df[i] = x_freq[i] * dpsih[i];
                }
                
                // Convert to FFT format
                for i in 0..pad_len {
                    dff[i] = FFTComplex::new(df[i].re, df[i].im);
                }
                
                // Transform back to time domain
                ifft.process(dff);
                
                // Normalize and store
                if let Some(ref mut dWx_arr) = dWx {
                    for j in 0..pad_len {
                        let val = Complex64::new(
                            dff[j].re * norm_factor,
                            dff[j].im * norm_factor
                        );
                        
                        // Apply L1/L2 normalization
                        dWx_arr[[scale_idx, j]] = if l1_norm {
                            val
                        } else {
                            val * scale.sqrt() // L2 norm
                        };
                    }
                }
            }
        }
    }
    
    (Wx, dWx)
}

// Optimized log scale generation
fn generate_log_scales(N: usize, nv: usize, wavelet: &str) -> PyResult<Array1<f64>> {
    // Determine min/max scales based on wavelet type
    let (min_scale, max_scale) = match wavelet {
        "gmw" => (2.0_f64, (N as f64) * 0.5),
        "morlet" => (2.0_f64, (N as f64) * 0.5),
        _ => (2.0_f64, (N as f64) * 0.5)
    };
    
    // Determine number of scales based on voices per octave
    let log_min = min_scale.log2();
    let log_max = max_scale.log2();
    let num_octaves = log_max - log_min;
    let num_scales = (num_octaves * nv as f64).ceil() as usize;
    
    // Generate scales using a more efficient approach
    let mut scales = Vec::with_capacity(num_scales);
    let scale_factor = if num_scales > 1 {
        (log_max - log_min) / ((num_scales - 1) as f64)
    } else {
        0.0
    };
    
    for i in 0..num_scales {
        let power = log_min + (i as f64) * scale_factor;
        scales.push(2.0_f64.powf(power));
    }
    
    Ok(Array1::from(scales))
}

/// Generate wavelets efficiently in Fourier domain
pub fn generate_wavelet_fourier(xi: &Array1<f64>, scale: f64, wavelet_type: &str) -> Array1<Complex64> {
    let N = xi.len();
    let mut psih = Array1::<Complex64>::zeros(N);
    
    match wavelet_type {
        "morlet" => {
            // Morlet parameters
            let mu = 6.0_f64; // Default value
            let neg_half = -0.5_f64;
            let norm = std::f64::consts::PI.powf(-0.25) * std::f64::consts::SQRT_2;
            let k_exp = (-0.5 * mu.powi(2)).exp();
            
            // Pre-compute for efficiency
            let scaled_mu = mu * scale;
            
            // Apply scale with SIMD-friendly approach
            for i in 0..N {
                let w = scale * xi[i];
                
                // Only fill positive frequencies for analytic wavelet
                if w >= 0.0 {
                    // Morlet formula (optimized)
                    let w_minus_mu = w - mu;
                    let term1 = (neg_half * w_minus_mu.powi(2)).exp();
                    let term2 = k_exp * (neg_half * w.powi(2)).exp();
                    
                    psih[i] = Complex64::new(norm * (term1 - term2), 0.0);
                }
            }
        },
        "gmw" | _ => {
            // GMW parameters
            let gamma = 3.0_f64;
            let beta = 60.0_f64;
            
            // Pre-compute constants
            let gamma_inv = 1.0 / gamma;
            let wc = (beta * gamma_inv).powf(gamma_inv);
            
            // Apply scale
            for i in 0..N {
                let w = scale * xi[i];
                
                // Only fill positive frequencies for analytic wavelet
                if w > 0.0 {
                    // L1-norm Generalized Morse Wavelet
                    let ln_w = w.ln();
                    let term = (beta * ln_w - w.powf(gamma)).exp();
                    psih[i] = Complex64::new(2.0 * term, 0.0);
                }
            }
        }
    }
    
    psih
}

/// Performs inverse CWT with optimizations
#[pyfunction]
#[pyo3(signature = (Wx, wavelet="gmw", scales=None, nv=None, one_int=true, x_len=None, x_mean=0.0, padtype="reflect", rpadded=false, l1_norm=true))]
pub fn icwt<'py>(
    py: Python<'py>,
    Wx: PyReadonlyArray2<Complex64>,
    wavelet: &str,
    scales: Option<PyReadonlyArray1<f64>>,
    nv: Option<usize>,
    one_int: bool,
    x_len: Option<usize>,
    x_mean: f64,
    padtype: &str,
    rpadded: bool,
    l1_norm: bool,
) -> PyResult<PyObject> {
    // Get dimensions
    let Wx_array = Wx.as_array();
    let (n_scales, n_times) = (Wx_array.shape()[0], Wx_array.shape()[1]);
    
    // Process scales
    let scales_array = match scales {
        Some(s) => s.as_array().to_owned(),
        None => {
            return Err(PyValueError::new_err("Scales must be provided"));
        }
    };
    
    // Calculate admissibility constant
    let adm_constant = match wavelet {
        "morlet" => 0.776, // Approximation for Morlet with mu=6
        "gmw" => 1.0,      // Approximation for GMW
        _ => 1.0
    };
    
    // Execute reconstruction
    let x_reconstructed = Python::allow_threads(py, || {
        let x_length = x_len.unwrap_or(n_times);
        
        if one_int {
            // One-integral method (simpler and often more accurate)
            let mut x = vec![0.0; x_length];
            
            // Calculate normalization constants first
            let dj = if n_scales > 1 && scales_array[1] > scales_array[0] {
                (scales_array[1] / scales_array[0]).ln()
            } else {
                0.1 // Default value
            };
            let final_norm = (2.0 / adm_constant) * dj;
            
            // Process scales in parallel
            // Pre-compute normalized contributions from each scale
            let scale_contributions: Vec<_> = (0..n_scales).into_par_iter()
                .map(|i| {
                    let scale = scales_array[i];
                    let norm_factor = if l1_norm {
                        1.0 
                    } else {
                        1.0 / scale.sqrt()
                    };
                    
                    let mut scale_contrib = vec![0.0; x_length];
                    for j in 0..x_length {
                        scale_contrib[j] = Wx_array[[i, j]].re * norm_factor;
                    }
                    scale_contrib
                })
                .collect();
            
            // Combine all contributions
            for j in 0..x_length {
                for i in 0..n_scales {
                    x[j] += scale_contributions[i][j];
                }
                x[j] = x[j] * final_norm + x_mean;
            }
            
            Array1::from(x)
        } else {
            // Two-integral method with optimizations
            let mut x = Array1::<f64>::zeros(x_length);
            
            // Get frequency grid once
            let xi = xifn(1.0, x_length);
            
            // Process scales in parallel using thread-local FFT planners
            let scale_results: Vec<_> = (0..n_scales).into_par_iter()
                .map(|i| {
                    let scale = scales_array[i];
                    
                    // Create a thread-local FFT planner
                    let mut planner = FftPlanner::new();
                    let fft = planner.plan_fft_forward(x_length);
                    let ifft = planner.plan_fft_inverse(x_length);
                    
                    // Create a buffer for this scale's result
                    let mut scale_result = vec![0.0; x_length];
                    
                    // Create wavelet at this scale
                    let psih = generate_wavelet_fourier(&xi, scale, wavelet);
                    
                    // Extract this scale's CWT coefficients
                    let mut tmp = vec![Complex64::new(0.0, 0.0); x_length];
                    for j in 0..x_length {
                        tmp[j] = Wx_array[[i, j]];
                    }
                    
                    // Convert to frequency domain
                    let mut tmp_fft = tmp.iter()
                        .map(|&c| FFTComplex::new(c.re, c.im))
                        .collect::<Vec<_>>();
                    
                    fft.process(&mut tmp_fft);
                    
                    // Multiply by wavelet
                    for j in 0..x_length {
                        let tmp_complex = Complex64::new(tmp_fft[j].re, tmp_fft[j].im);
                        tmp[j] = tmp_complex * psih[j].conj();
                    }
                    
                    // Back to time domain
                    let mut tmp_ifft = tmp.iter()
                        .map(|&c| FFTComplex::new(c.re, c.im))
                        .collect::<Vec<_>>();
                    
                    ifft.process(&mut tmp_ifft);
                    
                    // Process result
                    let norm_factor = 1.0 / (x_length as f64);
                    let scale_norm = if l1_norm {
                        1.0 / scale
                    } else {
                        1.0 / scale.sqrt().powi(2)
                    };
                    
                    for j in 0..x_length {
                        scale_result[j] = tmp_ifft[j].re * norm_factor * scale_norm;
                    }
                    
                    scale_result
                })
                .collect();
            
            // Combine results
            for j in 0..x_length {
                for i in 0..n_scales {
                    x[j] += scale_results[i][j];
                }
            }
            
            // Final normalization
            let dj = if n_scales > 1 && scales_array[1] > scales_array[0] {
                (scales_array[1] / scales_array[0]).ln()
            } else {
                0.1 // Default value
            };
            
            let norm = (2.0 / adm_constant) * dj;
            for j in 0..x_length {
                x[j] = x[j] * norm + x_mean;
            }
            
            x
        }
    });
    
    // Convert to Python array
    Ok(x_reconstructed.into_pyarray(py).to_object(py))
}