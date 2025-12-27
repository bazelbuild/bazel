"""Test file for Label() constructor with package-relative labels."""

# Test absolute label
_absolute_label = Label("//foo/bar:baz.bzl")

# Test package-relative label (no colon)
_relative_label = Label("test.bzl")

# Test package-relative label (with colon)
_relative_with_colon = Label(":test.bzl")

def test_function():
    """A function that uses Label constructor."""
    # Use the labels to verify they work
    _ = Label("another.bzl")
    _ = Label("//external:file.bzl")
    pass
