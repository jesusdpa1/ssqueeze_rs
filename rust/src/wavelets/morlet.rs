// rust/src/wavelets/morlet.rs
use ndarray::{Array1, ArrayView1};
use num_complex::Complex64;
use numpy::{IntoPyArray, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use std::f64::consts::PI;
use super::base::{WaveletBase, xifn};

/// Morlet wavelet implementation
pub struct Morlet {
    mu: f64,
    dtype: String,
}

impl Morlet {
    pub fn new(mu: f64, dtype: String) -> Self {
        Morlet { mu, dtype }
    }
    
    /// Helper function to compute Morlet wavelet constants
    fn compute_constants(&self) -> (f64, f64) {
        // Compute normalization constants
        let cs = (1.0 + (-self.mu.powi(2)).exp() - 2.0 * (-3.0/4.0 * self.mu.powi(2)).exp()).powf(-0.5);
        let ks = (-0.5 * self.mu.powi(2)).exp();
        
        (cs, ks)
    }
}

impl WaveletBase for Morlet {
    fn psih(&self, xi: &ArrayView1<f64>) -> Array1<Complex64> {
        let (cs, ks) = self.compute_constants();
        let factor = (2.0_f64).sqrt() * cs * PI.powf(0.25);
        
        let mut output = Array1::zeros(xi.len());
        for (i, &w) in xi.iter().enumerate() {
            // Apply the Morlet formula
            let term1 = (-0.5 * (w - self.mu).powi(2)).exp();
            let term2 = ks * (-0.5 * w.powi(2)).exp();
            output[i] = Complex64::new(factor * (term1 - term2), 0.0);
        }
        
        output
    }
    
    fn center_frequency(&self) -> f64 {
        // For Morlet, center frequency is approximately mu
        self.mu
    }
    
    fn name(&self) -> String {
        format!("Morlet (mu={})", self.mu)
    }
}

// Simple Morlet wrapper for Python

#[pyfunction]
#[pyo3(signature = (w, mu=6.0, dtype="float64"))]
pub fn morlet(
    py: Python<'_>,
    w: PyReadonlyArray1<f64>,
    mu: f64,
    dtype: &str,
) -> PyResult<PyObject> {
    // Create the Morlet wavelet
    let wavelet = Morlet::new(mu, dtype.to_string());
    
    // Compute the wavelet
    let w_array = w.as_array();
    let result = wavelet.psih(&w_array);
    
    // Convert to Python
    Ok(result.into_pyarray(py).into_py(py))
}

/// Compute the Morlet wavelet in frequency domain
#[pyfunction]
#[pyo3(signature = (n=1024, scale=1.0, mu=6.0, dtype="float64"))]
pub fn morlet_freq<'py>(
    py: Python<'py>,
    n: usize,
    scale: f64,
    mu: f64,
    dtype: &str,
) -> PyResult<PyObject> {
    // Generate the frequency grid
    let xi = xifn(scale, n);
    
    // Create the Morlet wavelet
    let wavelet = Morlet::new(mu, dtype.to_string());
    
    // Compute the wavelet
    let result = wavelet.psih(&xi.view());
    
    // Convert to Python
    Ok(result.into_pyarray(py).into_py(py))
}

/// Compute the Morlet wavelet in time domain via inverse FFT
#[pyfunction]
#[pyo3(signature = (n=1024, scale=1.0, mu=6.0, dtype="float64"))]
pub fn morlet_time<'py>(
    py: Python<'py>,
    n: usize,
    scale: f64,
    mu: f64,
    dtype: &str,
) -> PyResult<PyObject> {
    // Generate the frequency domain wavelet
    let xi = xifn(scale, n);
    let wavelet = Morlet::new(mu, dtype.to_string());
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