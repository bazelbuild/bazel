"""Input file to test angle bracket bug (https://github.com/bazelbuild/skydoc/issues/186)"""

def bracket_function(name):
    """Dummy docstring with <brackets>.

    This rule runs checks on <angle brackets>.

    Args:
        name: an arg with <b>formatted</b> docstring.

    Returns:
        some <angled> brackets

    """
    pass

bracketuse = provider(
    doc = "Information with <brackets>",
    fields = {
        "foo": "A string representing foo",
        "bar": "A string representing bar",
        "baz": "A string representing baz",
    },
)

def _rule_impl(ctx):
    return []

my_anglebrac = rule(
    implementation = _rule_impl,
    doc = "Rule with <brackets>",
    attrs = {
        "useless": attr.string(
            doc = "Args with some <b>formatting</b>",
            default = "Find brackets",
        ),
    },
)
