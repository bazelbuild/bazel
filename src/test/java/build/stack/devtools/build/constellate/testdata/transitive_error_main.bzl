"""Main file that loads from a file with errors.

This tests best-effort extraction when transitive loads have missing symbols.
"""

load("transitive_error_dep.bzl", "dep_function")

def main_function(x):
    """A function in the main file.

    Args:
        x: The input value

    Returns:
        The processed value
    """
    return dep_function(x)

# Define a rule in the main file
def _main_rule_impl(ctx):
    return []

main_rule = rule(
    implementation = _main_rule_impl,
    doc = "A rule in the main file.",
    attrs = {
        "value": attr.string(default = "default"),
    },
)
