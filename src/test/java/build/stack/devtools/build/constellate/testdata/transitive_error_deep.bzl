"""Deep dependency file that loads from a file with a missing symbol error.

This file loads from missing_symbol_file.bzl which has an error, but we want
to continue and make deep_function available despite that error.
"""

load("load_test_lib.bzl", "lib_function")
# This load will fail in best-effort mode, but we still want to define our function
load("missing_symbol_file.bzl", "some_function")

def deep_function(x):
    """A function defined in this file.

    Args:
        x: Input value

    Returns:
        Processed value
    """
    return lib_function(x)
