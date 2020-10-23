"""Input file for input template test"""

def template_function(foo):
    """Runs some checks on the given function parameter.

    This rule runs checks on a given function parameter in chosen template.
    Use `bazel build` to run the check.

    Args:
        foo: A unique name for this function.
    """
    pass

example = provider(
    doc = "Stores information about an example in chosen template.",
    fields = {
        "foo": "A string representing foo",
        "bar": "A string representing bar",
        "baz": "A string representing baz",
    },
)

def _rule_impl(ctx):
    return []

my_example = rule(
    implementation = _rule_impl,
    doc = "Small example of rule using chosen template.",
    attrs = {
        "useless": attr.string(
            doc = "This argument will be ignored.",
            default = "word",
        ),
    },
)

def my_aspect_impl(ctx):
    return []

my_aspect = aspect(
    implementation = my_aspect_impl,
    doc = "This is my aspect. It does stuff.",
    attr_aspects = ["deps", "attr_aspect"],
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
    },
)
