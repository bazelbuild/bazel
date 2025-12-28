"""Test file to demonstrate wrapper functions (traditional "macros").

This shows the pattern where a function wraps a rule call - what developers
traditionally call a "macro" before first-class macros were introduced.
"""

def _my_library_impl(ctx):
    """Implementation of my_library rule."""
    return []

my_library = rule(
    implementation = _my_library_impl,
    doc = "A library rule.",
    attrs = {
        "srcs": attr.label_list(doc = "Source files"),
        "deps": attr.label_list(doc = "Dependencies"),
        "visibility": attr.label_list(doc = "Visibility"),
    },
)

def my_library_macro(name, srcs = [], deps = [], **kwargs):
    """A wrapper function (traditional macro) that calls my_library.

    This is what most developers think of as a "macro" - a function that
    calls a rule and potentially does some preprocessing.

    Args:
        name: The name of the target
        srcs: Source files
        deps: Dependencies
        **kwargs: Additional arguments forwarded to my_library
    """
    # This function just forwards to the rule
    my_library(
        name = name,
        srcs = srcs,
        deps = deps,
        **kwargs
    )

def complex_macro(name, srcs = [], **kwargs):
    """A more complex wrapper that creates multiple targets.

    Args:
        name: The base name for targets
        srcs: Source files
        **kwargs: Additional arguments
    """
    # Creates multiple targets - classic macro behavior
    my_library(
        name = name + "_lib",
        srcs = srcs,
    )

    my_library(
        name = name,
        deps = [":" + name + "_lib"],
        **kwargs
    )

def helper_function(param):
    """A regular helper function that doesn't call rules.

    This should NOT be detected as a macro/wrapper.

    Args:
        param: Some parameter

    Returns:
        A processed value
    """
    return param.upper()
