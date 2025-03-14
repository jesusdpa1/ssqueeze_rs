// rust/src/lib.rs
use pyo3::prelude::*;

mod spectral;
mod utils;
mod wavelets;
mod ridge;

// Re-export the STFT functions from the spectral module
use spectral::stft::stft;
use spectral::ssq_stft::ssq_stft;
use spectral::cwt::cwt;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from ssqueeze!".to_string()
}

/// Python module entry point
#[pymodule]
fn _rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add standalone functions
    m.add_function(wrap_pyfunction!(hello_from_bin, py)?)?;
    
    // Add the STFT functions directly to the _rs module
    m.add_function(wrap_pyfunction!(stft, py)?)?;
    m.add_function(wrap_pyfunction!(ssq_stft, py)?)?;
    m.add_function(wrap_pyfunction!(cwt, py)?)?;
    
    Ok(())
}