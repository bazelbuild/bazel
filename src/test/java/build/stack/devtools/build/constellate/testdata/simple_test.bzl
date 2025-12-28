"""Simple test file for basic constellate functionality."""

def simple_function(x):
    """A simple function.

    Args:
        x: The input value

    Returns:
        The input value doubled
    """
    return x * 2

SimpleInfo = provider(
    doc = "A simple provider.",
    fields = ["value"],
)

def _simple_rule_impl(ctx):
    return [SimpleInfo(value = ctx.attr.value)]

simple_rule = rule(
    implementation = _simple_rule_impl,
    doc = "A simple rule.",
    attrs = {
        "value": attr.string(default = "default"),
    },
)
