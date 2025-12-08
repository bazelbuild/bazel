"""File that has a missing symbol error."""

# This will cause "does not contain symbol" error
load("load_test_lib.bzl", "MissingSymbol")

def some_function():
    """A function that won't be defined because the load fails."""
    return MissingSymbol()
