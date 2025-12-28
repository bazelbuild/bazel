"""Test file for verifying load statement location tracking.

This file contains multiple load statements at known line numbers
to verify that SymbolLocation data is correctly captured.
"""

# Load statement at line 8
load("load_test_lib.bzl", "lib_function", "LibInfo")

# Load statement at line 11
load("simple_test.bzl", "simple_rule")

# Load statement at line 14 with aliasing
load("load_test_lib.bzl", lib_func = "lib_function", lib_prov = "LibInfo", lib = "lib_rule")

def test_function():
    """A simple test function."""
    pass
