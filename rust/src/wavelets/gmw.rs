// rust/src/wavelets/gmw.rs
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f64::consts::PI;
use super::base::{WaveletBase, xifn};

/// Generalized Morse Wavelet implementation
pub struct GMW {
    gamma: f64,
    beta: f64,
    norm: String,
    order: i32,
    dtype: String,
}

impl GMW {
    pub fn new(gamma: f64, beta: f64, norm: String, order: i32, dtype: String) -> Self {
        GMW { 
            gamma, 
            beta, 
            norm: norm.to_lowercase(), 
            order, 
            dtype 
        }
    }
    
    /// Compute peak frequency for this GMW configuration
    pub fn peak_frequency(&self) -> f64 {
        (self.beta / self.gamma).powf(1.0 / self.gamma)
    }
    
    /// Calculate r parameter (used in multiple calculations)
    fn r_param(&self) -> f64 {
        (2.0 * self.beta + 1.0) / self.gamma
    }
    
    /// Calculate normalization constant for GMW
    fn normalization_constant(&self) -> f64 {
        let r = self.r_param();
        
        if self.norm == "bandpass" {
            // L1 normalization (peak value = 2)
            let wc = self.peak_frequency();
            2.0 / (self.beta * wc.ln() - wc.powf(self.gamma)).exp()
        } else {
            // L2 normalization (energy = 1)
            let gamma_r = gamma_function(r);
            (2.0 * PI * self.gamma * 2.0_f64.powf(r) / gamma_r).sqrt()
        }
    }
    
    /// Compute Laguerre polynomials for higher order GMWs
    fn laguerre_polynomial(&self, x: f64, k: i32, c: f64) -> f64 {
        let mut result = 0.0;
        
        for m in 0..=k {
            let binomial = binomial_coefficient(k + c as i32 + 1, c as i32 + m + 1) *
                           binomial_coefficient(k, m);
            let term = (-1.0_f64).powi(m) * x.powi(m) / factorial(m);
            result += binomial * term;
        }
        
        result
    }
}

impl WaveletBase for GMW {
    fn psih(&self, xi: &ArrayView1<f64>) -> Array1<Complex64> {
        let mut output = Array1::zeros(xi.len());
        
        if self.order == 0 {
            // Base GMW computation (more efficient)
            if self.norm == "bandpass" {
                // L1 normalization
                let wc = self.peak_frequency();
                let norm_const = self.normalization_constant();
                
                for (i, &w) in xi.iter().enumerate() {
                    if w <= 0.0 {
                        output[i] = Complex64::new(0.0, 0.0);
                        continue;
                    }
                    
                    let beta_term = self.beta * w.ln();
                    let gamma_term = w.powf(self.gamma);
                    let value = norm_const * (beta_term - gamma_term).exp();
                    output[i] = Complex64::new(value, 0.0);
                }
            } else {
                // L2 normalization
                let norm_const = self.normalization_constant();
                
                for (i, &w) in xi.iter().enumerate() {
                    if w <= 0.0 {
                        output[i] = Complex64::new(0.0, 0.0);
                        continue;
                    }
                    
                    let beta_term = w.powf(self.beta);
                    let gamma_term = (-w.powf(self.gamma)).exp();
                    let value = norm_const * beta_term * gamma_term;
                    output[i] = Complex64::new(value, 0.0);
                }
            }
        } else {
            // Higher-order GMW computation
            let r = self.r_param();
            let c = r - 1.0;
            let k = self.order;
            
            // Compute normalization coefficient
            let coeff = if self.norm == "bandpass" {
                let wc = self.peak_frequency();
                let norm_const = (gamma_function(r) * gamma_function(k as f64 + 1.0) /
                                gamma_function(k as f64 + r)).sqrt();
                2.0 * norm_const
            } else {
                (2.0 * PI * self.gamma * 2.0_f64.powf(r) *
                 gamma_function(k as f64 + 1.0) / gamma_function(k as f64 + r)).sqrt()
            };
            
            for (i, &w) in xi.iter().enumerate() {
                if w <= 0.0 {
                    output[i] = Complex64::new(0.0, 0.0);
                    continue;
                }
                
                // Compute the Laguerre polynomial term
                let laguerre_term = self.laguerre_polynomial(2.0 * w.powf(self.gamma), k, c);
                
                if self.norm == "bandpass" {
                    let wc = self.peak_frequency();
                    let beta_term = self.beta * w.ln();
                    let gamma_term = w.powf(self.gamma);
                    let exp_term = (-self.beta * wc.ln() + wc.powf(self.gamma) + 
                                   beta_term - gamma_term).exp();
                    
                    let value = coeff * laguerre_term * exp_term;
                    output[i] = Complex64::new(value, 0.0);
                } else {
                    let beta_term = w.powf(self.beta);
                    let gamma_term = (-w.powf(self.gamma)).exp();
                    
                    let value = coeff * laguerre_term * beta_term * gamma_term;
                    output[i] = Complex64::new(value, 0.0);
                }
            }
        }
        
        output
    }
    
    fn center_frequency(&self) -> f64 {
        self.peak_frequency()
    }
    
    fn name(&self) -> String {
        format!("GMW (γ={}, β={}, order={})", self.gamma, self.beta, self.order)
    }
}

/// Gamma function approximation
fn gamma_function(x: f64) -> f64 {
    // Use Lanczos approximation for Gamma function
    if x < 0.5 {
        let pi = std::f64::consts::PI;
        pi / ((pi * x).sin() * gamma_function(1.0 - x))
    } else {
        let p = [
            676.5203681218851,
            -1259.1392167224028,
            771.32342877765313,
            -176.61502916214059,
            12.507343278686905,
            -0.13857109526572012,
            9.9843695780195716e-6,
            1.5056327351493116e-7
        ];
        
        let x = x - 1.0;
        let mut y = 0.99999999999980993;
        for i in 0..p.len() {
            y += p[i] / (x + (i as f64) + 1.0);
        }
        
        let t = x + (p.len() as f64) - 0.5;
        let sqrt_2pi = (2.0 * std::f64::consts::PI).sqrt();
        sqrt_2pi * t.powf(x + 0.5) * (-t).exp() * y
    }
}

/// Calculate factorial: n!
fn factorial(n: i32) -> f64 {
    if n <= 1 {
        return 1.0;
    }
    (1..=n).map(|i| i as f64).product()
}

/// Calculate binomial coefficient
fn binomial_coefficient(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    if k == 0 || k == n {
        return 1.0;
    }
    
    // Compute directly for small values to avoid numeric issues
    if n <= 20 {
        return factorial(n) / (factorial(k) * factorial(n - k));
    }
    
    // Use logarithms for larger values to avoid overflow
    let mut c = 0.0;
    for i in 1..=k {
        c += ((n - k + i) as f64).ln() - (i as f64).ln();
    }
    c.exp()
}

/// Python interface for GMW wavelet
#[pyfunction]
#[pyo3(signature = (w, gamma=3.0, beta=60.0, norm="bandpass", order=0, dtype="float64"))]
pub fn gmw(
    py: Python<'_>,
    w: PyReadonlyArray1<f64>,
    gamma: f64,
    beta: f64,
    norm: &str,
    order: i32,
    dtype: &str,
) -> PyResult<PyObject> {
    // Validate parameters
    if gamma <= 0.0 {
        return Err(PyValueError::new_err("gamma must be positive"));
    }
    if beta < 0.0 {
        return Err(PyValueError::new_err("beta must be non-negative"));
    }
    if order < 0 {
        return Err(PyValueError::new_err("order must be non-negative"));
    }
    
    // Create the wavelet
    let wavelet = GMW::new(gamma, beta, norm.to_string(), order, dtype.to_string());
    
    // Compute the wavelet
    let w_array = w.as_array();
    let result = wavelet.psih(&w_array);
    
    // Convert to Python
    Ok(result.into_pyarray(py).into_py(py))
}

/// Compute the GMW wavelet in frequency domain
#[pyfunction]
#[pyo3(signature = (n=1024, scale=1.0, gamma=3.0, beta=60.0, norm="bandpass", order=0, dtype="float64"))]
pub fn gmw_freq<'py>(
    py: Python<'py>,
    n: usize,
    scale: f64,
    gamma: f64,
    beta: f64,
    norm: &str,
    order: i32,
    dtype: &str,
) -> PyResult<PyObject> {
    // Generate the frequency grid
    let xi = xifn(scale, n);
    
    // Create the GMW wavelet
    let wavelet = GMW::new(gamma, beta, norm.to_string(), order, dtype.to_string());
    
    // Compute the wavelet
    let result = wavelet.psih(&xi.view());
    
    // Convert to Python
    Ok(result.into_pyarray(py).into_py(py))
}

/// Compute the GMW wavelet in time domain via inverse FFT
#[pyfunction]
#[pyo3(signature = (n=1024, scale=1.0, gamma=3.0, beta=60.0, norm="bandpass", order=0, dtype="float64"))]
pub fn gmw_time<'py>(
    py: Python<'py>,
    n: usize,
    scale: f64,
    gamma: f64,
    beta: f64,
    norm: &str,
    order: i32,
    dtype: &str,
) -> PyResult<PyObject> {
    // Generate the frequency domain wavelet
    let xi = xifn(scale, n);
    let wavelet = GMW::new(gamma, beta, norm.to_string(), order, dtype.to_string());
    let mut psih = wavelet.psih(&xi.view());
    
    // Apply spectral reversal for proper time-domain centering
    let pn: Array1<f64> = Array1::from_iter((0..n).map(|i| (-1.0_f64).powi(i as i32)));
    
    for i in 0..n {
        psih[i] *= pn[i];
    }
    
    // If even length, halve the Nyquist bin
    if n % 2 == 0 {
        psih[n/2] /= 2.0;
    }
    
    // Perform inverse FFT
    let mut planner = rustfft::FftPlanner::new();
    let fft = planner.plan_fft_inverse(n);
    
    let mut fft_data: Vec<rustfft::num_complex::Complex<f64>> = psih.iter()
        .map(|&c| rustfft::num_complex::Complex::new(c.re, c.im))
        .collect();
    
    fft.process(&mut fft_data);
    
    // Convert back to ndarray
    let scale_factor = 1.0 / (n as f64);
    let psi: Array1<Complex64> = Array1::from_iter(
        fft_data.iter().map(|&c| Complex64::new(c.re * scale_factor, c.im * scale_factor))
    );
    
    // Convert to Python
    Ok(psi.into_pyarray(py).into_py(py))
}

/// Calculate GMW wavelet center frequency
#[pyfunction]
#[pyo3(signature = (gamma=3.0, beta=60.0, kind="peak"))]
pub fn gmw_center_frequency(
    gamma: f64,
    beta: f64,
    kind: &str,
) -> PyResult<f64> {
    match kind {
        "peak" => {
            // Simple peak frequency calculation
            Ok((beta / gamma).powf(1.0 / gamma))
        },
        "energy" => {
            // Energy frequency calculation
            let r = (2.0 * beta + 1.0) / gamma;
            let energy_freq = (1.0 / 2.0_f64.powf(1.0 / gamma)) * 
                             (gamma_function((2.0 * beta + 2.0) / gamma) / 
                              gamma_function((2.0 * beta + 1.0) / gamma));
            Ok(energy_freq)
        },
        _ => Err(PyValueError::new_err(format!("Unknown center frequency kind: {}", kind))),
    }
}