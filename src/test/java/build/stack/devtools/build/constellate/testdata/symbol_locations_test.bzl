# Test file for verifying symbol locations are collected for all entity types

# PROVIDER - Line 3
MyProvider = provider(
    doc = "A test provider",
    fields = ["value"],
)

# FUNCTION - Line 9
def my_function(name):
    """A test function."""
    pass

# RULE - Line 14
def _my_rule_impl(ctx):
    pass

my_rule = rule(
    implementation = _my_rule_impl,
    doc = "A test rule",
    attrs = {
        "srcs": attr.label_list(),
    },
)

# ASPECT - Line 26
def _my_aspect_impl(target, ctx):
    pass

my_aspect = aspect(
    implementation = _my_aspect_impl,
    doc = "A test aspect",
)

# MACRO - Line 35
def my_macro(name, visibility = None):
    """A test macro."""
    my_rule(name = name, visibility = visibility)
