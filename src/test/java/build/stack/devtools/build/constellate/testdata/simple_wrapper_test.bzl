"""Simple wrapper function test without kwargs to avoid macro resolution conflicts."""

def _my_rule_impl(ctx):
    """Implementation function."""
    return []

my_rule = rule(
    implementation = _my_rule_impl,
    doc = "A test rule.",
    attrs = {
        "srcs": attr.label_list(doc = "Sources"),
    },
)

def wrapper_without_kwargs(name, srcs):
    """A wrapper that calls a rule but doesn't use **kwargs.

    This won't trigger the macro resolution system.

    Args:
        name: Target name
        srcs: Source files
    """
    my_rule(
        name = name,
        srcs = srcs,
    )

def helper_func(value):
    """A helper that doesn't call any rules.

    Args:
        value: Some value

    Returns:
        Processed value
    """
    return value.strip()
