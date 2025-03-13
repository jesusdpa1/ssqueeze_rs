// rust/src/lib.rs
use pyo3::prelude::*;

mod spectral;

#[pyfunction]
fn hello_from_bin() -> String {
    "Hello from ssqueeze!".to_string()
}


/// A Python module implemented in Rust. The name of this function must match
/// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
/// import the module.
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    numpy::initialize_numpy();
    m.add_function(wrap_pyfunction!(hello_from_bin, m)?)?;
        // Add submodules
    m.add_submodule(spectral::create_module(_py)?)?;
    
    Ok(())
}
