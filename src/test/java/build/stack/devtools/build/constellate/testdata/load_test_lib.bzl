"""Library file for testing cross-file load and OriginKey tracking."""

def lib_function(x):
    """A function defined in the library file.

    Args:
        x: The input value

    Returns:
        The value multiplied by 3
    """
    return x * 3

LibInfo = provider(
    doc = "A provider defined in the library file.",
    fields = ["lib_value"],
)

def _lib_rule_impl(ctx):
    return [LibInfo(lib_value = ctx.attr.value)]

lib_rule = rule(
    implementation = _lib_rule_impl,
    doc = "A rule defined in the library file.",
    attrs = {
        "value": attr.string(default = "lib_default"),
    },
)
