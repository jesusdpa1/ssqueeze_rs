// rust/src/lib.rs
use pyo3::prelude::*;

mod spectral;

// Re-export the STFT function from the spectral module
use spectral::stft::stft;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from ssqueeze!".to_string()
}

/// Python module entry point
#[pymodule]
fn _rs(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Add standalone functions
    m.add_function(wrap_pyfunction!(hello_from_bin, py)?)?;
    
    // Add the STFT function directly to the _rs module
    m.add_function(wrap_pyfunction!(stft, py)?)?;
    
    Ok(())
}