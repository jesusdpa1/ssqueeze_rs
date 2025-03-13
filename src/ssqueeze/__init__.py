# src/ssqueeze/__init__.py
try:
    from . import _rs
except ImportError:
    import sys

    print(
        "Failed to import Rust module '_rs'. Make sure it's properly compiled.",
        file=sys.stderr,
    )

    # Create a dummy module for development purposes
    class _RsDummy:
        def hello_from_bin(self):
            return "Dummy Rust module (not compiled)"

        def stft(self, x, n_fft, hop_length, window, padtype):
            raise NotImplementedError("Rust module not compiled")

    _rs = _RsDummy()

# Expose the module
__all__ = ["_rs"]


def main():
    print(_rs.hello_from_bin())
