"""Test file to demonstrate name parameter forwarding patterns.

This tests tracking of the 'name' parameter as it gets forwarded from
wrapper functions to the underlying rules/macros.
"""

def _my_rule_impl(ctx):
    """Implementation of my_rule."""
    return []

my_rule = rule(
    implementation = _my_rule_impl,
    doc = "A test rule.",
    attrs = {
        "srcs": attr.label_list(doc = "Source files"),
        "deps": attr.label_list(doc = "Dependencies"),
    },
)

def _my_binary_impl(ctx):
    """Implementation of my_binary."""
    return []

my_binary = rule(
    implementation = _my_binary_impl,
    doc = "A binary rule.",
    attrs = {
        "srcs": attr.label_list(doc = "Source files"),
    },
)

def explicit_name_macro(name, srcs = [], **kwargs):
    """Macro that explicitly forwards the name parameter.

    This is the most common pattern - the name parameter is explicitly
    passed to the underlying rule with name=name.

    Args:
        name: The target name
        srcs: Source files
        **kwargs: Additional arguments
    """
    my_rule(
        name = name,
        srcs = srcs,
        **kwargs
    )

def positional_name_macro(name, srcs = []):
    """Macro that forwards name as a positional argument.

    Less common but still valid pattern.

    Args:
        name: The target name
        srcs: Source files
    """
    my_rule(name, srcs = srcs)

def transformed_name_macro(name, **kwargs):
    """Macro that transforms the name before forwarding.

    Common pattern where the name is modified (e.g., adding suffix).
    This should still be detected as forwarding the name parameter.

    Args:
        name: The base target name
        **kwargs: Additional arguments
    """
    my_rule(
        name = name + "_lib",
        **kwargs
    )

def hardcoded_name_macro(name, **kwargs):
    """Macro that doesn't actually forward the name parameter.

    The name parameter is present but not used - a hardcoded value is used instead.
    This should NOT be detected as forwarding.

    Args:
        name: The target name (unused)
        **kwargs: Additional arguments
    """
    my_rule(
        name = "hardcoded_target",
        **kwargs
    )

def multiple_name_macro(name, srcs = [], **kwargs):
    """Macro that creates multiple targets, forwarding name to multiple rules.

    Args:
        name: The base name for targets
        srcs: Source files
        **kwargs: Additional arguments
    """
    # First target - forwards name directly
    my_rule(
        name = name,
        srcs = srcs,
    )

    # Second target - uses name in transformation
    my_binary(
        name = name + "_binary",
        srcs = srcs,
        **kwargs
    )

def no_name_param_macro(srcs = [], **kwargs):
    """Macro that doesn't have a name parameter at all.

    Args:
        srcs: Source files
        **kwargs: Additional arguments (but name is not a separate param)
    """
    my_rule(
        name = "fixed_name",
        srcs = srcs,
        **kwargs
    )

def name_from_kwargs_macro(srcs = [], **kwargs):
    """Macro where name comes from kwargs, not explicit parameter.

    This is less common but demonstrates that name can come from **kwargs.

    Args:
        srcs: Source files
        **kwargs: Additional arguments including name
    """
    my_rule(
        srcs = srcs,
        **kwargs  # name is in kwargs
    )

def name_without_kwargs_macro(name, srcs = []):
    """Macro that forwards name but not kwargs.

    Shows explicit name forwarding without **kwargs forwarding.

    Args:
        name: The target name
        srcs: Source files
    """
    my_rule(
        name = name,
        srcs = srcs,
    )
