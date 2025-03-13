// rust/src/spectral/mod.rs
use pyo3::prelude::*;

pub mod stft;

/// Create and return the spectral module
pub fn create_module(py: Python) -> PyResult<&PyModule> {
    let module = PyModule::new(py, "spectral")?;
    
    module.add_function(wrap_pyfunction!(stft::stft, module)?)?;
    module.add_function(wrap_pyfunction!(stft::ssq_stft, module)?)?;
    
    Ok(module)
}