// rust/src/spectral/cwt.rs
use ndarray::{Array1, Array2, ArrayView1, Axis, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use crate::utils::array::{pad_signal, next_power_of_2};
use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};

/// Compute the continuous wavelet transform
/// 
/// # Arguments
/// * `x` - Input signal (1D array)
/// * `wavelet_type` - Type of wavelet to use (e.g. "morlet", "gmw")
/// * `scales` - Scales at which to compute the CWT
/// * `fs` - Sampling frequency (default: 1.0)
/// * `padtype` - Type of padding to use (default: "reflect")
/// * `l1_norm` - Whether to use L1 normalization (default: true)
/// * `derivative` - Whether to also compute the derivative of the CWT (default: false)
/// 
/// # Returns
/// * `Wx` - CWT of x
/// * `scales` - Scales at which CWT was computed
/// * `dWx` - Time-derivative of CWT of x, if requested
#[pyfunction]
#[pyo3(signature = (x, wavelet_type="morlet", mu=5.0, scales=None, nv=32, fs=1.0, padtype="reflect", l1_norm=true, derivative=false))]
pub fn cwt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    wavelet_type: &str,
    mu: f64,
    scales: Option<PyReadonlyArray1<f64>>,
    nv: usize,
    fs: f64,
    padtype: &str,
    l1_norm: bool,
    derivative: bool,
) -> PyResult<(PyObject, PyObject, Option<PyObject>)> {
    // Convert Python arrays to Rust arrays
    let x_array = x.as_array().to_owned();
    let N = x_array.len();
    
    // Process scales
    let scales_array = if let Some(scales) = scales {
        scales.as_array().to_owned()
    } else {
        // Generate log scales if not provided
        generate_default_scales(N, nv, wavelet_type, mu)?
    };
    
    // Allow Python threads to run during computation
    let (Wx, dWx) = Python::allow_threads(py, || {
        // Pad signal to next power of 2
        let N_padded = next_power_of_2(N);
        let padded_x = match padtype {
            "reflect" => pad_signal_reflect(&x_array, N_padded),
            "zero" => pad_signal_zero(&x_array, N_padded),
            _ => pad_signal_reflect(&x_array, N_padded),
        };
        
        // Pre-compute frequency domain
        let omega = generate_frequency_domain(N_padded);
        
        // Convert signal to frequency domain
        let mut planner = FftPlanner::new();
        let fft_forward = planner.plan_fft_forward(N_padded);
        
        let mut x_freq: Vec<FFTComplex<f64>> = padded_x.iter()
            .map(|&x| FFTComplex::new(x, 0.0))
            .collect();
        
        fft_forward.process(&mut x_freq);
        
        // Convert to ndarray
        let x_freq_array: Array1<Complex64> = x_freq.iter()
            .map(|&c| Complex64::new(c.re, c.im))
            .collect();
        
        // Compute CWT for each scale in parallel
        let num_scales = scales_array.len();
        let Wx_results: Vec<_> = (0..num_scales)
            .into_par_iter()
            .map(|i| {
                let scale = scales_array[i];
                
                // Generate wavelet in frequency domain at current scale
                let psih = generate_wavelet_fourier(&omega, scale, wavelet_type, mu);
                
                // Multiply signal FFT by wavelet
                let mut product = Array1::<Complex64>::zeros(N_padded);
                for j in 0..N_padded {
                    product[j] = x_freq_array[j] * psih[j];
                }
                
                // Convert back to Vec<FFTComplex> for inverse FFT
                let mut product_fft: Vec<FFTComplex<f64>> = product.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                
                // Create inverse FFT planner
                let mut planner = FftPlanner::new();
                let fft_inverse = planner.plan_fft_inverse(N_padded);
                
                // Perform inverse FFT
                fft_inverse.process(&mut product_fft);
                
                // Normalize
                let scale_factor = 1.0 / (N_padded as f64);
                let cwt_scale: Vec<Complex64> = product_fft.iter()
                    .map(|&c| Complex64::new(c.re * scale_factor, c.im * scale_factor))
                    .collect();
                
                // Apply L1 normalization if requested
                let normalized: Vec<Complex64> = if l1_norm {
                    cwt_scale.iter().map(|&c| c / scale.sqrt()).collect()
                } else {
                    cwt_scale
                };
                
                (i, normalized)
            })
            .collect();
        
        // Compute derivative if requested
        let dWx_results = if derivative {
            Some((0..num_scales)
                .into_par_iter()
                .map(|i| {
                    let scale = scales_array[i];
                    
                    // Generate derivative wavelet in frequency domain
                    let dpsih = generate_derivative_wavelet(&omega, scale, wavelet_type, mu);
                    
                    // Multiply signal FFT by derivative wavelet
                    let mut product = Array1::<Complex64>::zeros(N_padded);
                    for j in 0..N_padded {
                        product[j] = x_freq_array[j] * dpsih[j] * fs; // Scale by fs
                    }
                    
                    // Convert back to Vec<FFTComplex> for inverse FFT
                    let mut product_fft: Vec<FFTComplex<f64>> = product.iter()
                        .map(|&c| FFTComplex::new(c.re, c.im))
                        .collect();
                    
                    // Create inverse FFT planner
                    let mut planner = FftPlanner::new();
                    let fft_inverse = planner.plan_fft_inverse(N_padded);
                    
                    // Perform inverse FFT
                    fft_inverse.process(&mut product_fft);
                    
                    // Normalize
                    let scale_factor = 1.0 / (N_padded as f64);
                    let dcwt_scale: Vec<Complex64> = product_fft.iter()
                        .map(|&c| Complex64::new(c.re * scale_factor, c.im * scale_factor))
                        .collect();
                    
                    // Apply L1 normalization if requested
                    let normalized: Vec<Complex64> = if l1_norm {
                        dcwt_scale.iter().map(|&c| c / scale.sqrt()).collect()
                    } else {
                        dcwt_scale
                    };
                    
                    (i, normalized)
                })
                .collect()
            )
        } else {
            None
        };
        
        // Assemble results
        let mut Wx = Array2::<Complex64>::zeros((num_scales, N));
        let mut dWx = if derivative {
            Some(Array2::<Complex64>::zeros((num_scales, N)))
        } else {
            None
        };
        
        // Unpad the results
        let start_idx = (N_padded - N) / 2;
        
        // Fill Wx with the results
        for (i, scale_result) in Wx_results {
            for j in 0..N {
                Wx[[i, j]] = scale_result[start_idx + j];
            }
        }
        
        // Fill dWx with results if derivative was computed
        if let Some(dWx_results) = dWx_results {
            let dWx_arr = dWx.as_mut().unwrap();
            for (i, scale_result) in dWx_results {
                for j in 0..N {
                    dWx_arr[[i, j]] = scale_result[start_idx + j];
                }
            }
        }
        
        (Wx, dWx)
    });
    
    // Convert results back to Python objects
    let py_Wx = Wx.into_pyarray(py).to_object(py);
    let py_scales = scales_array.into_pyarray(py).to_object(py);
    let py_dWx = match dWx {
        Some(d) => Some(d.into_pyarray(py).to_object(py)),
        None => None,
    };
    
    Ok((py_Wx, py_scales, py_dWx))
}

/// Generate default logarithmic scales for CWT
fn generate_default_scales(N: usize, nv: usize, wavelet_type: &str, mu: f64) -> PyResult<Array1<f64>> {
    // Compute min/max scales
    let (min_scale, max_scale) = match wavelet_type {
        "morlet" => {
            // For Morlet, minimum scale depends on mu
            // In general, min_scale ≈ 1/mu, max_scale ≈ N/4
            let min_scale = 1.0 / mu;
            let max_scale = (N as f64) / 4.0;
            (min_scale, max_scale)
        },
        "gmw" => {
            // GMW (Generalized Morse Wavelets) typical bounds
            let min_scale = 1.0;
            let max_scale = (N as f64) / 4.0;
            (min_scale, max_scale)
        },
        _ => {
            // Default values for other wavelets
            let min_scale = 1.0;
            let max_scale = (N as f64) / 4.0;
            (min_scale, max_scale)
        }
    };
    
    // Generate logarithmic scales
    let log_min = min_scale.ln();
    let log_max = max_scale.ln();
    let num_scales = nv * ((log_max - log_min).abs() / std::f64::consts::LN_2).ceil() as usize;
    
    let mut scales = Array1::<f64>::zeros(num_scales);
    let step = (log_max - log_min) / ((num_scales - 1) as f64);
    
    for i in 0..num_scales {
        scales[i] = (log_min + (i as f64) * step).exp();
    }
    
    Ok(scales)
}

/// Generate frequency domain for FFT
fn generate_frequency_domain(N: usize) -> Array1<f64> {
    let mut omega = Array1::<f64>::zeros(N);
    
    // Positive frequencies: [0, 1, ..., N/2]
    for i in 0..N/2+1 {
        omega[i] = 2.0 * std::f64::consts::PI * (i as f64) / (N as f64);
    }
    
    // Negative frequencies: [-N/2+1, ..., -1]
    for i in N/2+1..N {
        omega[i] = 2.0 * std::f64::consts::PI * ((i as f64) - (N as f64)) / (N as f64);
    }
    
    omega
}

/// Generate Morlet wavelet in Fourier domain
fn generate_wavelet_fourier(omega: &Array1<f64>, scale: f64, wavelet_type: &str, mu: f64) -> Array1<Complex64> {
    let N = omega.len();
    let mut psih = Array1::<Complex64>::zeros(N);
    
    match wavelet_type {
        "morlet" => {
            let scaled_omega = scale * omega;
            
            // Morlet wavelet formula
            for i in 0..N {
                let w = scaled_omega[i];
                let exp_term = (-0.5 * (w - mu).powi(2)).exp();
                let correction = (-0.5 * mu.powi(2)).exp(); // Admissibility correction
                
                // Analytic wavelet: only positive frequencies
                if w >= 0.0 {
                    psih[i] = Complex64::new(exp_term - correction, 0.0);
                }
            }
        },
        "gmw" => {
            // Simplified GMW implementation (for more complete one, see the GMW module)
            let gamma = 3.0; // Default gamma parameter
            let beta = 60.0;  // Default beta parameter
            
            // Calculate peak frequency
            let wc = (beta / gamma).powf(1.0 / gamma);
            
            let scaled_omega = scale * omega;
            for i in 0..N {
                let w = scaled_omega[i];
                
                // Only positive frequencies for analytic wavelet
                if w > 0.0 {
                    let term = 2.0 * (w.powf(beta) * (-w.powf(gamma)).exp());
                    psih[i] = Complex64::new(term, 0.0);
                }
            }
        },
        _ => {
            // Default to Morlet if unknown wavelet type
            let scaled_omega = scale * omega;
            
            for i in 0..N {
                let w = scaled_omega[i];
                let exp_term = (-0.5 * (w - mu).powi(2)).exp();
                let correction = (-0.5 * mu.powi(2)).exp();
                
                if w >= 0.0 {
                    psih[i] = Complex64::new(exp_term - correction, 0.0);
                }
            }
        }
    }
    
    psih
}

/// Generate derivative of wavelet in Fourier domain (for time-derivative of CWT)
fn generate_derivative_wavelet(omega: &Array1<f64>, scale: f64, wavelet_type: &str, mu: f64) -> Array1<Complex64> {
    let N = omega.len();
    let mut dpsih = Array1::<Complex64>::zeros(N);
    
    // Generate wavelet
    let psih = generate_wavelet_fourier(omega, scale, wavelet_type, mu);
    
    // Frequency-domain differentiation: multiply by i*omega
    for i in 0..N {
        let w = omega[i];
        dpsih[i] = psih[i] * Complex64::new(0.0, w);
    }
    
    dpsih
}

/// Pad signal with reflection at boundaries
fn pad_signal_reflect(x: &Array1<f64>, pad_len: usize) -> Array1<Complex64> {
    let n = x.len();
    
    // Simple case: input already matches desired length
    if n == pad_len {
        return x.iter().map(|&val| Complex64::new(val, 0.0)).collect();
    }
    
    let pad_size = pad_len - n;
    let pad_left = pad_size / 2;
    let pad_right = pad_size - pad_left;
    
    let mut padded = Array1::<Complex64>::zeros(pad_len);
    
    // Copy original signal to the middle
    for i in 0..n {
        padded[i + pad_left] = Complex64::new(x[i], 0.0);
    }
    
    // Reflect on the left
    for i in 0..pad_left {
        let mirror_idx = pad_left - i;
        if mirror_idx < n {
            padded[i] = Complex64::new(x[mirror_idx], 0.0);
        }
    }
    
    // Reflect on the right
    for i in 0..pad_right {
        let mirror_idx = n - 2 - i;
        if mirror_idx >= 0 && mirror_idx < n {
            padded[n + pad_left + i] = Complex64::new(x[mirror_idx], 0.0);
        }
    }
    
    padded
}

/// Pad signal with zeros
fn pad_signal_zero(x: &Array1<f64>, pad_len: usize) -> Array1<Complex64> {
    let n = x.len();
    
    // Simple case: input already matches desired length
    if n == pad_len {
        return x.iter().map(|&val| Complex64::new(val, 0.0)).collect();
    }
    
    let pad_size = pad_len - n;
    let pad_left = pad_size / 2;
    
    let mut padded = Array1::<Complex64>::zeros(pad_len);
    
    // Copy original signal to the middle
    for i in 0..n {
        padded[i + pad_left] = Complex64::new(x[i], 0.0);
    }
    
    padded
}

/// Inverse CWT - reconstruct signal from wavelet transform
#[pyfunction]
#[pyo3(signature = (Wx, scales, wavelet_type="morlet", mu=5.0, l1_norm=true))]
pub fn icwt<'py>(
    py: Python<'py>,
    Wx: PyReadonlyArray1<Complex64>,
    scales: PyReadonlyArray1<f64>,
    wavelet_type: &str, 
    mu: f64,
    l1_norm: bool,
) -> PyResult<PyObject> {
    // Convert Python arrays to Rust arrays
    let Wx_array = Wx.as_array().to_owned();
    let scales_array = scales.as_array().to_owned();
    
    // Extract dimensions
    let num_scales = scales_array.len();
    let n = Wx_array.shape()[1];
    
    // Calculate admissibility constant
    let adm_coef = match wavelet_type {
        "morlet" => calculate_admissibility_morlet(mu),
        "gmw" => calculate_admissibility_gmw(3.0, 60.0), // Default gamma=3, beta=60
        _ => calculate_admissibility_morlet(mu), // Default to Morlet
    };
    
    // Allow Python threads to run during computation
    let x = Python::allow_threads(py, || {
        // Compute reconstructed signal by summing over scales
        let mut x = Array1::<f64>::zeros(n);
        
        // Compute scale step (for logarithmic scales)
        let scale_step = if num_scales > 1 {
            (scales_array[1] / scales_array[0]).ln()
        } else {
            1.0
        };
        
        // Sum over all scales
        for i in 0..num_scales {
            let scale = scales_array[i];
            
            // Extract current scale from Wx and take real part
            let mut scale_contrib = Array1::<f64>::zeros(n);
            for j in 0..n {
                scale_contrib[j] = Wx_array[[i, j]].re;
            }
            
            // Apply normalization
            let norm_factor = if l1_norm {
                // L1 norm - divide by scale
                1.0 / scale
            } else {
                // L2 norm - divide by sqrt(scale)
                1.0 / scale.sqrt()
            };
            
            // Add contribution to result
            for j in 0..n {
                x[j] += scale_contrib[j] * norm_factor;
            }
        }
        
        // Apply normalization constants
        let C = 2.0 / adm_coef;
        let dj = scale_step;
        
        for j in 0..n {
            x[j] *= C * dj;
        }
        
        x
    });
    
    // Convert result back to Python
    let py_x = x.into_pyarray(py).to_object(py);
    
    Ok(py_x)
}

/// Calculate admissibility coefficient for Morlet wavelet
fn calculate_admissibility_morlet(mu: f64) -> f64 {
    // Integration approximation for Morlet wavelet's admissibility constant
    // This is a simplified version - for exact calculation, numerical integration is needed
    let sigma = 1.0;
    let pi = std::f64::consts::PI;
    
    let correction = (-0.5 * mu.powi(2)).exp();
    
    // Approximation based on analytical formula
    pi.sqrt() * sigma * (1.0 - correction.powi(2))
}

/// Calculate admissibility coefficient for GMW
fn calculate_admissibility_gmw(gamma: f64, beta: f64) -> f64 {
    // Integration approximation for GMW admissibility constant
    // This is an approximation - for exact calculation, numerical integration is needed
    let pi = std::f64::consts::PI;
    
    // Gamma function approximation for (2*beta + 1) / gamma
    let r = (2.0 * beta + 1.0) / gamma;
    let gamma_r = gamma_approximation(r);
    
    2.0 * pi * gamma * 2.0_f64.powf(r) / gamma_r
}

/// Simple approximation of gamma function for GMW admissibility
fn gamma_approximation(x: f64) -> f64 {
    if x == 1.0 {
        return 1.0;
    }
    if x == 2.0 {
        return 1.0;
    }
    if x == 0.5 {
        return std::f64::consts::PI.sqrt();
    }
    
    // Stirling's approximation for gamma function
    let e = std::f64::consts::E;
    let pi = std::f64::consts::PI;
    
    (2.0 * pi / x).sqrt() * (x / e).powf(x)
}