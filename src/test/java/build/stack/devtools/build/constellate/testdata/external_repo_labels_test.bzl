"""Test file for verifying external repository label parsing.

This file contains load statements from external repositories to verify
that the Label proto correctly captures repo, pkg, and name fields.
"""

# External repo with file in root package
load("@bazel_features//:features.bzl", "bazel_features")

# External repo with file in nested package
load("@rules_go//go:def.bzl", "go_library", "go_test")

# External repo with different package structure
load("@io_bazel_rules_docker//container:container.bzl", "container_image")

# Local (current repo) load for comparison
load("//src/test/java/build/stack/devtools/build/constellate/testdata:load_test_lib.bzl", "lib_function")

def test_function():
    """Function to verify external loads work."""
    pass
