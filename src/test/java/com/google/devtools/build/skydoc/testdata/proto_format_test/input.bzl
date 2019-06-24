"""Input file for proto format test"""

def check_function(foo):
    """Runs some checks on the given function parameter.

    This rule runs checks on a given function parameter.
    Use `bazel build` to run the check.

    Args:
        foo: A unique name for this rule.
    """
    pass

example = provider(
    doc = "Stores information about an example.",
    fields = {
        "foo": "A string representing foo",
        "bar": "A string representing bar",
        "baz": "A string representing baz",
    },
)

def _rule_impl(ctx):
    print("Hello World")

my_example = rule(
    implementation = _rule_impl,
    doc = "Small example of rule.",
    attrs = {
        "useless": attr.string(
            doc = "This argument will be ignored.",
            default = "ignoreme",
        ),
    },
)
