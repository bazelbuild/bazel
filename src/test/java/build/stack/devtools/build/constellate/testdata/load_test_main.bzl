"""Main file that loads from another file to test cross-file OriginKey tracking."""

load("load_test_lib.bzl", "lib_function", "LibInfo", "lib_rule")

def main_function(x):
    """A function in the main file that uses loaded function.

    Args:
        x: The input value

    Returns:
        The value processed by lib_function
    """
    return lib_function(x)

# Re-export loaded entities to test OriginKey preservation
exported_lib_function = lib_function
exported_lib_info = LibInfo
exported_lib_rule = lib_rule

# Define entities in the main file
MainInfo = provider(
    doc = "A provider defined in the main file.",
    fields = ["main_value"],
)

def _main_rule_impl(ctx):
    return [MainInfo(main_value = ctx.attr.value), LibInfo(lib_value = "from_main")]

main_rule = rule(
    implementation = _main_rule_impl,
    doc = "A rule in the main file that provides both MainInfo and LibInfo.",
    attrs = {
        "value": attr.string(default = "main_default"),
    },
    provides = [MainInfo],
)
