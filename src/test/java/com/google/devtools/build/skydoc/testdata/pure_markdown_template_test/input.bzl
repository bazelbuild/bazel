"""Input file for markdown template test"""

def example_function(foo, bar = "bar"):
    """Small example of function using a markdown template.

    Args:
        foo: This parameter does foo related things.
        bar: This parameter does bar related things.
    """
    pass

ExampleProviderInfo = provider(
    doc = "Small example of provider using a markdown template.",
    fields = {
        "foo": "A string representing foo",
        "bar": "A string representing bar",
        "baz": "A string representing baz",
    },
)

def _rule_impl(ctx):
    return []

example_rule = rule(
    implementation = _rule_impl,
    doc = "Small example of rule using a markdown template.",
    attrs = {
        "first": attr.string(doc = "This is the first attribute"),
        "second": attr.string(default = "2"),
    },
)

def _aspect_impl(ctx):
    return []

example_aspect = aspect(
    implementation = _aspect_impl,
    doc = "Small example of aspect using a markdown template.",
    attr_aspects = ["deps", "attr_aspect"],
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string(doc = "This is the second attribute."),
    },
)
