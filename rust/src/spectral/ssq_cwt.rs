// rust/src/spectral/ssq_cwt.rs
use ndarray::{Array1, Array2, s};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use rayon::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex as FFTComplex};
use crate::utils::array::{pad_reflect, pad_zero, next_power_of_2};
use crate::wavelets::base::xifn;
use crate::spectral::cwt::generate_wavelet_fourier;
use std::f64::consts::PI;

/// Compute the phase transform for CWT
fn phase_cwt(
    Wx: &Array2<Complex64>,
    dWx: &Array2<Complex64>,
    gamma: f64,
) -> Array2<f64> {
    let (n_scales, n_times) = (Wx.shape()[0], Wx.shape()[1]);
    let mut w = Array2::<f64>::zeros((n_scales, n_times));
    
    // Compute phase transform in parallel by scales
    w.axis_iter_mut(ndarray::Axis(0))
        .into_par_iter()
        .enumerate()
        .for_each(|(i, mut w_row)| {
            for j in 0..n_times {
                if Wx[[i, j]].norm() < gamma {
                    w_row[j] = f64::INFINITY;
                } else {
                    // Calculate phase transform using the formula:
                    // w[a,b] = Im((1/2pi) * (1/Wx[a,b]) * d/db (Wx[a,b]))
                    let a = dWx[[i, j]].re;
                    let b = dWx[[i, j]].im;
                    let c = Wx[[i, j]].re;
                    let d = Wx[[i, j]].im;
                    
                    // Use optimized formula to avoid complex division
                    let phase_derivative = (b * c - a * d) / ((c * c + d * d) * 2.0 * PI);
                    w_row[j] = phase_derivative.abs();
                }
            }
        });
    
    w
}

/// Compute frequencies associated with CWT scales
fn compute_associated_frequencies(
    n_freqs: usize,
    min_freq: f64,
    max_freq: f64,
    distribution: &str,
) -> Array1<f64> {
    match distribution {
        "log" => {
            // Logarithmic distribution
            let log_min = min_freq.log2();
            let log_max = max_freq.log2();
            
            // Generate frequencies using a more efficient approach
            let mut freqs = Vec::with_capacity(n_freqs);
            let scale_factor = if n_freqs > 1 {
                (log_max - log_min) / ((n_freqs - 1) as f64)
            } else {
                0.0
            };
            
            for i in 0..n_freqs {
                let power = log_min + (i as f64) * scale_factor;
                freqs.push(2.0_f64.powf(power));
            }
            
            Array1::from(freqs)
        },
        "linear" => {
            // Linear distribution
            let mut freqs = Vec::with_capacity(n_freqs);
            let step = if n_freqs > 1 {
                (max_freq - min_freq) / ((n_freqs - 1) as f64)
            } else {
                0.0
            };
            
            for i in 0..n_freqs {
                freqs.push(min_freq + (i as f64) * step);
            }
            
            Array1::from(freqs)
        },
        _ => {
            // Default to logarithmic
            let log_min = min_freq.log2();
            let log_max = max_freq.log2();
            
            // Generate frequencies using a more efficient approach
            let mut freqs = Vec::with_capacity(n_freqs);
            let scale_factor = if n_freqs > 1 {
                (log_max - log_min) / ((n_freqs - 1) as f64)
            } else {
                0.0
            };
            
            for i in 0..n_freqs {
                let power = log_min + (i as f64) * scale_factor;
                freqs.push(2.0_f64.powf(power));
            }
            
            Array1::from(freqs)
        }
    }
}

/// Perform synchrosqueezing of CWT
fn ssqueeze(
    Wx: &Array2<Complex64>,
    w: &Array2<f64>,
    ssq_freqs: &Array1<f64>,
    scales: &Array1<f64>,
    gamma: f64,
    squeezing: &str,
    flipud: bool,
) -> Array2<Complex64> {
    // Note: scales and gamma are currently unused in this implementation
    let _ = scales;
    let _ = gamma;
    let (n_scales, n_times) = (Wx.shape()[0], Wx.shape()[1]);
    let n_freqs = ssq_freqs.len();
    
    // Initialize output array
    let mut Tx = Array2::<Complex64>::zeros((n_freqs, n_times));
    
    // Calculate frequency bin size for constant in synchrosqueezing
    let is_log = if n_freqs > 1 {
        ssq_freqs[1] / ssq_freqs[0] > 1.1 // Detect if frequencies are logarithmically spaced
    } else {
        false
    };
    
    // Determine binning parameters for frequency mapping
    let (log_min, log_step, lin_min, lin_step) = if is_log {
        let log_min = ssq_freqs[0].log2();
        let log_step = if n_freqs > 1 {
            (ssq_freqs[n_freqs-1].log2() - log_min) / ((n_freqs - 1) as f64)
        } else {
            1.0
        };
        (log_min, log_step, 0.0, 0.0)
    } else {
        let lin_min = ssq_freqs[0];
        let lin_step = if n_freqs > 1 {
            (ssq_freqs[n_freqs-1] - lin_min) / ((n_freqs - 1) as f64)
        } else {
            1.0
        };
        (0.0, 0.0, lin_min, lin_step)
    };
    
    // Process synchrosqueezing in parallel by time columns
    // Collect results from parallel computation
    let columns: Vec<_> = (0..n_times).into_par_iter().map(|j| {
        let mut Tx_col = vec![Complex64::new(0.0, 0.0); n_freqs];
        
        for i in 0..n_scales {
            // Skip inf/nan values
            if w[[i, j]].is_infinite() || w[[i, j]].is_nan() {
                continue;
            }
            
            // Find closest frequency bin
            let k = if is_log {
                // Logarithmic binning
                let w_value = w[[i, j]];
                let log_w = w_value.log2();
                let bin = ((log_w - log_min) / log_step).round() as isize;
                if bin < 0 || bin >= n_freqs as isize {
                    continue;
                }
                if flipud {
                    (n_freqs - 1) - bin as usize
                } else {
                    bin as usize
                }
            } else {
                // Linear binning
                let bin = ((w[[i, j]] - lin_min) / lin_step).round() as isize;
                if bin < 0 || bin >= n_freqs as isize {
                    continue;
                }
                if flipud {
                    (n_freqs - 1) - bin as usize
                } else {
                    bin as usize
                }
            };
            
            // Apply synchrosqueezing
            let weight = match squeezing {
                "sum" => Wx[[i, j]],
                "lebesgue" => {
                    // Each coefficient contributes the same amount
                    Complex64::new(1.0 / (n_scales as f64), 0.0)
                },
                _ => Wx[[i, j]], // Default to sum
            };
            
            Tx_col[k] += weight;
        }
        
        (j, Tx_col)
    }).collect();
    
    // Copy columns to the output array
    for (j, col) in columns {
        for k in 0..n_freqs {
            Tx[[k, j]] = col[k];
        }
    }
    
    Tx
}

/// Synchrosqueezed Continuous Wavelet Transform
/// 
/// # Arguments
/// * `x` - Input signal (1D array)
/// * `wavelet` - Wavelet function or predefined wavelet name
/// * `scales` - Scales at which to compute the CWT
/// * `fs` - Sampling frequency
/// * `t` - Time vector
/// * `ssq_freqs` - Frequencies to synchrosqueeze CWT scales onto
/// * `nv` - Number of voices per octave
/// * `padtype` - Type of padding to use
/// * `squeezing` - Synchrosqueezing method ('sum' or 'lebesgue')
/// * `maprange` - Frequency mapping range ('maximal', 'peak', 'energy')
/// * `difftype` - Differentiation type ('trig', 'phase', 'numeric')
/// * `gamma` - CWT phase threshold
/// * `flipud` - Flip frequency axis
/// 
/// # Returns
/// * `Tx` - Synchrosqueezed CWT of x
/// * `ssq_freqs` - Frequencies associated with rows of Tx
#[pyfunction]
#[pyo3(signature = (
    x, 
    wavelet="gmw", 
    scales=None, 
    fs=None, 
    t=None, 
    ssq_freqs=None,
    nv=32, 
    padtype="reflect", 
    squeezing="sum", 
    maprange="peak",
    difftype="trig", 
    gamma=None, 
    vectorized=true,
    flipud=true
))]
pub fn ssq_cwt<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<f64>,
    wavelet: &str,
    scales: Option<PyReadonlyArray1<f64>>,
    fs: Option<f64>,
    t: Option<PyReadonlyArray1<f64>>,
    ssq_freqs: Option<&str>,
    nv: usize,
    padtype: &str,
    squeezing: &str,
    maprange: &str,
    difftype: &str,
    gamma: Option<f64>,
    vectorized: bool,
    flipud: bool
) -> PyResult<(PyObject, PyObject)> {
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
    
    // Note unused parameters to avoid compiler warnings
    let _ = difftype;  // Currently, only 'trig' method is implemented
    let _ = vectorized; // Currently always using parallel approach
    
    // Process scales
    let scales_array = match scales {
        Some(s) => s.as_array().to_owned(),
        None => {
            // Generate logarithmic scales (similar to generate_log_scales in cwt.rs)
            let min_scale = 2.0_f64;
            let max_scale = (N as f64) * 0.5;
            
            let log_min = min_scale.log2();
            let log_max = max_scale.log2();
            let num_octaves = log_max - log_min;
            let num_scales = (num_octaves * nv as f64).ceil() as usize;
            
            let mut scales_vec = Vec::with_capacity(num_scales);
            let scale_factor = if num_scales > 1 {
                (log_max - log_min) / ((num_scales - 1) as f64)
            } else {
                0.0
            };
            
            for i in 0..num_scales {
                let power = log_min + (i as f64) * scale_factor;
                scales_vec.push(2.0_f64.powf(power));
            }
            
            Array1::from(scales_vec)
        }
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
        
        let x_freq = Array1::<Complex64>::from_iter(
            x_fft.iter().map(|&c| Complex64::new(c.re, c.im))
        );
        
        // Calculate padding indices for unpadding later
        let n1 = (pad_len - N) / 2;
        
        // Compute frequency grid
        let xi = xifn(1.0, pad_len);
        
        let num_scales = scales_array.len();
        
        // Initialize CWT results
        let mut Wx = Array2::<Complex64>::zeros((num_scales, pad_len));
        let mut dWx = Array2::<Complex64>::zeros((num_scales, pad_len));
        
        // Compute CWT and derivative in parallel using a thread-safe approach
        let scale_results: Vec<_> = scales_array.iter().enumerate()
            .map(|(i, &scale)| (i, scale))
            .collect::<Vec<_>>()
            .par_iter()
            .map(|&(i, scale)| {
                // Generate wavelet
                let psih = generate_wavelet_fourier(&xi, scale, wavelet);
                
                // Generate derivative wavelet
                let mut dpsih = psih.clone();
                for (j, &w) in xi.iter().enumerate() {
                    dpsih[j] *= Complex64::new(0.0, w / dt);
                }
                
                // Thread-local FFT planner
                let mut planner = FftPlanner::new();
                let ifft = planner.plan_fft_inverse(pad_len);
                
                // Convolution in frequency domain
                let mut Wx_freq = Vec::with_capacity(pad_len);
                let mut dWx_freq = Vec::with_capacity(pad_len);
                
                for j in 0..pad_len {
                    Wx_freq.push(x_freq[j] * psih[j]);
                    dWx_freq.push(x_freq[j] * dpsih[j]);
                }
                
                // Convert to FFT format
                let mut Wx_fft: Vec<FFTComplex<f64>> = Wx_freq.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                let mut dWx_fft: Vec<FFTComplex<f64>> = dWx_freq.iter()
                    .map(|&c| FFTComplex::new(c.re, c.im))
                    .collect();
                
                // Inverse FFT
                ifft.process(&mut Wx_fft);
                ifft.process(&mut dWx_fft);
                
                // Normalize and store results
                let norm_factor = 1.0 / (pad_len as f64);
                let mut row_results = Vec::with_capacity(pad_len);
                let mut drow_results = Vec::with_capacity(pad_len);
                
                for j in 0..pad_len {
                    row_results.push(Complex64::new(
                        Wx_fft[j].re * norm_factor,
                        Wx_fft[j].im * norm_factor
                    ));
                    drow_results.push(Complex64::new(
                        dWx_fft[j].re * norm_factor,
                        dWx_fft[j].im * norm_factor
                    ));
                }
                
                // Return the scale index and computed values
                (i, row_results, drow_results)
            })
            .collect();
            
        // Now combine the results into Wx and dWx arrays
        for (i, row_data, drow_data) in scale_results {
            for j in 0..pad_len {
                Wx[[i, j]] = row_data[j];
                dWx[[i, j]] = drow_data[j];
            }
        }
        
        // Unpad Wx and dWx
        let Wx_unpadded = Wx.slice(s![.., n1..(n1 + N)]).to_owned();
        let dWx_unpadded = dWx.slice(s![.., n1..(n1 + N)]).to_owned();
        
        // Set gamma if not provided
        let gamma_value = gamma.unwrap_or_else(|| {
            // Use epsilon based on data type, 10 * EPS64
            10.0 * 2.2204460492503131e-16 
        });
        
        // Compute phase transform
        let w = phase_cwt(&Wx_unpadded, &dWx_unpadded, gamma_value);
        
        // Compute frequencies for synchrosqueezed transform
        let ssq_freqs_dist = ssq_freqs.unwrap_or("log");
        
        // Determine frequency range based on the scales
        let (min_freq, max_freq) = match maprange {
            "maximal" => {
                // Use fundamental frequency (1/dT) and Nyquist (fs/2)
                let dT = N as f64 * dt;
                (1.0 / dT, 0.5 / dt)
            },
            "peak" | "energy" | _ => {
                // Use frequencies from min/max scales
                // Approximate conversion: freq ~ 1/scale
                (1.0 / scales_array[scales_array.len()-1], 1.0 / scales_array[0])
            }
        };
        
        // Generate SSQ frequencies
        let ssq_freqs_array = compute_associated_frequencies(
            num_scales, 
            min_freq, 
            max_freq, 
            ssq_freqs_dist
        );
        
        // Apply synchrosqueezing
        let Tx = ssqueeze(
            &Wx_unpadded, 
            &w, 
            &ssq_freqs_array, 
            &scales_array, 
            gamma_value, 
            squeezing,
            flipud
        );
        
        (Tx, ssq_freqs_array)
    });
    
    // Convert results to Python objects
    let (Tx, ssq_freqs) = result;
    
    // Using IntoPy trait instead of deprecated to_object method
    let py_Tx = Tx.into_pyarray(py).into_py(py);
    let py_ssq_freqs = ssq_freqs.into_pyarray(py).into_py(py);
    
    Ok((py_Tx, py_ssq_freqs))
}