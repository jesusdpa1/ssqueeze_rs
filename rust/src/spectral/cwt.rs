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
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(pad_len);
        
        let mut x_fft: Vec<FFTComplex<f64>> = padded_x.iter()
            .map(|&x| FFTComplex::new(x, 0.0))
            .collect();
        
        fft.process(&mut x_fft);
        
        // Convert to ndarray for easier manipulation
        let x_freq = Array1::<Complex64>::from_iter(
            x_fft.iter().map(|&c| Complex64::new(c.re, c.im))
        );
        
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

/// Compute CWT using vectorized approach (process all scales at once)
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
    
    // Compute frequency domain points
    let xi = xifn(1.0, pad_len);
    
    // Process all scales in parallel
    let results: Vec<_> = scales.iter().enumerate().collect::<Vec<_>>()
        .par_iter()
        .map(|&(scale_idx, &scale)| {
            // Generate wavelet at this scale
            let psih = generate_wavelet_fourier(&xi, scale, wavelet);
            
            // For derivative computation
            let dpsih = if derivative {
                // For frequency domain differentiation
                let mut dpsih = psih.clone();
                for (i, &w) in xi.iter().enumerate() {
                    dpsih[i] *= Complex64::new(0.0, w / dt);
                }
                Some(dpsih)
            } else {
                None
            };
            
            // Compute convolution in frequency domain (just element-wise multiply)
            let mut Wx_scale = Array1::<Complex64>::zeros(pad_len);
            let mut dWx_scale = if derivative {
                Some(Array1::<Complex64>::zeros(pad_len))
            } else {
                None
            };
            
            for i in 0..pad_len {
                Wx_scale[i] = x_freq[i] * psih[i];
                if let Some(ref mut dWx_s) = dWx_scale {
                    if let Some(ref dp) = dpsih {
                        dWx_s[i] = x_freq[i] * dp[i];
                    }
                }
            }
            
            // Transform back to time domain
            let mut planner = FftPlanner::new();
            let ifft = planner.plan_fft_inverse(pad_len);
            
            // For Wx
            let mut Wx_fft: Vec<FFTComplex<f64>> = Wx_scale.iter()
                .map(|&c| FFTComplex::new(c.re, c.im))
                .collect();
            
            ifft.process(&mut Wx_fft);
            
            // Normalize
            let norm_factor = 1.0 / (pad_len as f64);
            let Wx_time: Vec<Complex64> = Wx_fft.iter()
                .map(|&c| Complex64::new(c.re * norm_factor, c.im * norm_factor))
                .collect();
            
            // For dWx if needed
            let dWx_time = if let Some(dWx_s) = dWx_scale {
                let mut dWx_fft: Vec<FFTComplex<f64>> = dWx_s.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                
                ifft.process(&mut dWx_fft);
                
                // Normalize
                let dWx_t: Vec<Complex64> = dWx_fft.iter()
                    .map(|&c| Complex64::new(c.re * norm_factor, c.im * norm_factor))
                    .collect();
                
                Some(dWx_t)
            } else {
                None
            };
            
            // Apply L1 normalization if requested
            let Wx_norm = if l1_norm {
                // L1 norm - scale doesn't change energy distribution
                Wx_time
            } else {
                // L2 norm - scale affects energy
                Wx_time.iter().map(|&c| c * scale.sqrt()).collect()
            };
            
            let dWx_norm = if let Some(dWx_t) = dWx_time {
                if l1_norm {
                    Some(dWx_t)
                } else {
                    Some(dWx_t.iter().map(|&c| c * scale.sqrt()).collect())
                }
            } else {
                None
            };
            
            (scale_idx, Wx_norm, dWx_norm)
        })
        .collect();
    
    // Assemble results
    for (scale_idx, Wx_scale, dWx_scale) in results {
        for j in 0..pad_len {
            Wx[[scale_idx, j]] = Wx_scale[j];
            if let Some(ref mut dWx_arr) = dWx {
                if let Some(ref dWx_s) = dWx_scale {
                    dWx_arr[[scale_idx, j]] = dWx_s[j];
                }
            }
        }
    }
    
    (Wx, dWx)
}

// Fix for compute_cwt_loop to use xi.clone()
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
    
    // Process each scale
    for (scale_idx, &scale) in scales.iter().enumerate() {
        // Generate wavelet at this scale - use reference instead of moving xi
        let psih = generate_wavelet_fourier(&xi, scale, wavelet);
        
        // Compute convolution in frequency domain
        let mut Wx_freq = Array1::<Complex64>::zeros(pad_len);
        for i in 0..pad_len {
            Wx_freq[i] = x_freq[i] * psih[i];
        }
        
        // Transform back to time domain
        let mut planner = rustfft::FftPlanner::new();
        let ifft = planner.plan_fft_inverse(pad_len);
        
        let mut Wx_fft: Vec<rustfft::num_complex::Complex<f64>> = Wx_freq.iter()
            .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
            .collect();
        
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
            let mut dpsih = psih.clone();
            for (i, &w) in xi.iter().enumerate() {
                dpsih[i] *= Complex64::new(0.0, w / dt);
            }
            
            // Compute convolution
            let mut dWx_freq = Array1::<Complex64>::zeros(pad_len);
            for i in 0..pad_len {
                dWx_freq[i] = x_freq[i] * dpsih[i];
            }
            
            // Transform back to time domain
            let mut dWx_fft: Vec<rustfft::num_complex::Complex<f64>> = dWx_freq.iter()
                .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
                .collect();
            
            ifft.process(&mut dWx_fft);
            
            // Normalize and store
            if let Some(ref mut dWx_arr) = dWx {
                for j in 0..pad_len {
                    let val = Complex64::new(
                        dWx_fft[j].re * norm_factor,
                        dWx_fft[j].im * norm_factor
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
    
    (Wx, dWx)
}

// Fix for log scale generation
fn generate_log_scales(N: usize, nv: usize, wavelet: &str) -> PyResult<Array1<f64>> {
    // Determine min/max scales based on wavelet type
    // These values are simplified approximations - 
    // actual implementations often use more complex formulas
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
    
    // Generate scales
    let mut scales = Array1::<f64>::zeros(num_scales);
    for i in 0..num_scales {
        let power = log_min + (i as f64) * (log_max - log_min) / ((num_scales - 1) as f64);
        scales[i] = 2.0_f64.powf(power);
    }
    
    Ok(scales)
}

/// Generate Morlet wavelet in Fourier domain
// Update the function signature to take a reference instead of ownership
fn generate_wavelet_fourier(xi: &Array1<f64>, scale: f64, wavelet_type: &str) -> Array1<Complex64> {
    let N = xi.len();
    let mut psih = Array1::<Complex64>::zeros(N);
    
    match wavelet_type {
        "morlet" => {
            // Morlet parameters
            let mu = 6.0_f64; // Default value
            
            // Apply scale
            for i in 0..N {
                let w = scale * xi[i];
                
                // Only fill positive frequencies for analytic wavelet
                if w >= 0.0 {
                    // Morlet formula
                    let term1 = (-0.5 * (w - mu).powi(2)).exp();
                    let term2 = (-0.5 * mu.powi(2)).exp() * (-0.5 * w.powi(2)).exp();
                    
                    // Normalization constant
                    let norm = std::f64::consts::PI.powf(-0.25) * std::f64::consts::SQRT_2;
                    
                    psih[i] = Complex64::new(norm * (term1 - term2), 0.0);
                }
            }
        },
        "gmw" | _ => {
            // GMW parameters
            let gamma = 3.0_f64;
            let beta = 60.0_f64;
            
            // Calculate peak frequency (center frequency)
            let wc = (beta / gamma).powf(1.0 / gamma);
            
            // Apply scale
            for i in 0..N {
                let w = scale * xi[i];
                
                // Only fill positive frequencies for analytic wavelet
                if w > 0.0 {
                    // L1-norm Generalized Morse Wavelet
                    let term = (beta * w.ln() - w.powf(gamma)).exp();
                    psih[i] = Complex64::new(2.0 * term, 0.0);
                }
            }
        }
    }
    
    psih
}

/// Performs inverse CWT
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
            let mut x = Array1::<f64>::zeros(x_length);
            
            for i in 0..n_scales {
                let scale = scales_array[i];
                
                // Extract scale's contribution (real part)
                for j in 0..x_length {
                    let norm_factor = if l1_norm {
                        1.0 // Already L1 normalized
                    } else {
                        1.0 / scale.sqrt() // Convert from L2 to L1
                    };
                    
                    x[j] += Wx_array[[i, j]].re * norm_factor;
                }
            }
            
            // Calculate scale step
            let dj = if n_scales > 1 && scales_array[1] > scales_array[0] {
                (scales_array[1] / scales_array[0]).ln()
            } else {
                0.1 // Default value
            };
            
            // Apply final normalization
            let norm = (2.0 / adm_constant) * dj;
            for j in 0..x_length {
                x[j] *= norm;
                x[j] += x_mean; // Add mean (CWT removes DC component)
            }
            
            x
        } else {
            // Two-integral method
            // This implementation is simplified and may not be accurate for all wavelets
            // For a complete implementation, reference the Python code
            let mut x = Array1::<f64>::zeros(x_length);
            
            // Process each scale
            for i in 0..n_scales {
                let scale = scales_array[i];
                
                // Create wavelet at this scale
                let xi = xifn(scale, x_length);
                let psih = generate_wavelet_fourier(&xi, scale, wavelet);
                
                // Perform convolution in frequency domain
                let mut tmp = Array1::<Complex64>::zeros(x_length);
                for j in 0..x_length {
                    tmp[j] = Wx_array[[i, j]];
                }
                
                // Convert to frequency domain
                let mut planner = FftPlanner::new();
                let fft = planner.plan_fft_forward(x_length);
                
                let mut tmp_fft: Vec<FFTComplex<f64>> = tmp.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                
                fft.process(&mut tmp_fft);
                
                // Multiply by wavelet
                for j in 0..x_length {
                    let tmp_complex = Complex64::new(tmp_fft[j].re, tmp_fft[j].im);
                    tmp[j] = tmp_complex * psih[j].conj();
                }
                
                // Back to time domain
                let ifft = planner.plan_fft_inverse(x_length);
                
                let mut tmp_ifft: Vec<FFTComplex<f64>> = tmp.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                
                ifft.process(&mut tmp_ifft);
                
                // Add to result
                let norm_factor = 1.0 / (x_length as f64);
                for j in 0..x_length {
                    let val = tmp_ifft[j].re * norm_factor;
                    let scale_norm = if l1_norm {
                        1.0 / scale
                    } else {
                        1.0 / scale.sqrt().powi(2)
                    };
                    
                    x[j] += val * scale_norm;
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
                x[j] *= norm;
                x[j] += x_mean;
            }
            
            x
        }
    });
    
    // Convert to Python array
    Ok(x_reconstructed.into_pyarray(py).to_object(py))
}