"""Dependency file that tries to load a missing symbol from its own dependency.

This file successfully exports dep_function, but it loads from another file that has an error.
"""

# Load from a file that has a missing symbol error
load("transitive_error_deep.bzl", "deep_function")

def dep_function(x):
    """A function that uses a symbol from a deeper dependency.

    Args:
        x: Input value

    Returns:
        Processed value
    """
    # Try to use the deep_function (which may be a stub if the load failed)
    return deep_function(x)
