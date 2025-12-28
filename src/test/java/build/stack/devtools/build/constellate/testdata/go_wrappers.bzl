"""Wrapper functions that forward to private rules."""

load("go_binary.bzl", "go_binary", "go_non_executable_binary")

_SELECT_TYPE = type(select({"//conditions:default": ""}))

LINKMODE_NORMAL = "normal"
LINKMODES_EXECUTABLE = ["normal", "pie"]

def go_binary_macro(name, **kwargs):
    """See docs/go/core/rules.md#go_binary for full documentation.

    This macro wraps the private go_binary rule and handles platform-specific logic.

    Args:
        name: Name of the binary target
        **kwargs: Additional arguments passed to the underlying rule
    """
    if kwargs.get("goos") != None or kwargs.get("goarch") != None:
        for key, value in kwargs.items():
            if type(value) == _SELECT_TYPE:
                # In the long term, we should replace goos/goarch with Bazel-native platform
                # support, but while we have the mechanisms, we try to avoid people trying to use
                # _both_ goos/goarch _and_ native platform support.
                #
                # It's unclear to users whether the select should happen before or after the
                # goos/goarch is reconfigured, and we can't interpose code to force either
                # behaviour, so we forbid this.
                fail("Cannot use select for go_binary with goos/goarch set, but {} was a select".format(key))

    if kwargs.get("linkmode", LINKMODE_NORMAL) in LINKMODES_EXECUTABLE:
        go_binary(name = name, **kwargs)
    else:
        go_non_executable_binary(name = name, **kwargs)
