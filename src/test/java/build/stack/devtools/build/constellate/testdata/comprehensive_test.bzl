"""Comprehensive test file for constellate tool.

Tests all supported entity types with OriginKey, advertised providers, and provider init.
"""

# Simple provider with init callback and schema
MyInfoProvider = provider(
    doc = "A test provider with init callback and schema.",
    fields = {
        "value": "The value field",
        "count": "The count field",
    },
    init = lambda value, count = 0: {"value": value, "count": count},
)

# Provider without init
SimpleProvider = provider(
    doc = "A simple provider without init.",
    fields = ["data"],
)

def _my_rule_impl(ctx):
    """Implementation of my_rule."""
    return [MyInfoProvider(value = ctx.attr.value, count = 1)]

# Rule with advertised providers
my_rule = rule(
    implementation = _my_rule_impl,
    doc = "A test rule that provides MyInfoProvider.",
    attrs = {
        "value": attr.string(
            doc = "The value to store",
            default = "default",
        ),
        "deps": attr.label_list(
            doc = "Dependencies",
            providers = [MyInfoProvider],
        ),
    },
    provides = [MyInfoProvider],
)

def _my_aspect_impl(target, ctx):
    """Implementation of my_aspect."""
    return [SimpleProvider(data = "aspect data")]

# Aspect definition
my_aspect = aspect(
    implementation = _my_aspect_impl,
    doc = "A test aspect.",
    attr_aspects = ["deps"],
    attrs = {
        "aspect_param": attr.string(
            doc = "An aspect parameter",
            default = "param",
        ),
    },
)

def _my_repo_rule_impl(ctx):
    """Implementation of my_repo_rule."""
    ctx.file("BUILD", "# Generated BUILD file")

# Repository rule
my_repo_rule = repository_rule(
    implementation = _my_repo_rule_impl,
    doc = "A test repository rule.",
    attrs = {
        "url": attr.string(
            doc = "The URL to fetch",
            mandatory = True,
        ),
    },
    environ = ["MY_ENV_VAR"],
)

# Symbolic macro (requires --experimental_enable_first_class_macros)
def _my_macro_impl(name, value):
    """Implementation of my_macro."""
    native.filegroup(
        name = name,
        srcs = [],
        tags = [value],
    )

my_macro = macro(
    implementation = _my_macro_impl,
    doc = "A test symbolic macro.",
    attrs = {
        "value": attr.string(
            doc = "A value parameter",
            default = "default",
        ),
    },
)

# Module extension (requires Bzlmod)
def _my_extension_impl(ctx):
    """Implementation of my_extension."""
    pass

_tag = tag_class(
    doc = "A test tag class.",
    attrs = {
        "name": attr.string(
            doc = "The name attribute",
            mandatory = True,
        ),
        "version": attr.string(
            doc = "The version attribute",
            default = "1.0",
        ),
    },
)

my_extension = module_extension(
    implementation = _my_extension_impl,
    doc = "A test module extension.",
    tag_classes = {
        "install": _tag,
    },
)

# Function with comprehensive docstring
def my_function(param1, param2 = None, *args, **kwargs):
    """A test function with comprehensive documentation.

    This function demonstrates various parameter types and docstring formatting.

    Args:
        param1: The first required parameter
        param2: An optional parameter with default
        *args: Variable positional arguments
        **kwargs: Variable keyword arguments

    Returns:
        A dictionary with processed values

    Deprecated:
        Use new_function() instead
    """
    return {"param1": param1, "param2": param2}

# Nested in struct
test_struct = struct(
    nested_provider = provider(
        doc = "A provider nested in a struct.",
        fields = ["nested_value"],
    ),
    nested_function = lambda x: x * 2,
)
