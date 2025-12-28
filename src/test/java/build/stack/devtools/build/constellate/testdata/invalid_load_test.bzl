"""Test file with an invalid load statement."""

# This load statement has an invalid target name containing ':'
load(":cache.bzl", "some_symbol")

def test_function():
    """A test function."""
    pass
