"""Test file that uses select to verify builtin functions work correctly."""

# Test select at module load time (like rules_go does)
_SELECT_TYPE = type(select({"//conditions:default": ""}))

def function_with_select(name, srcs = []):
    """A function that uses select.

    Args:
        name: The name
        srcs: Source files (can use select)

    Returns:
        A configuration with platform-specific values
    """
    return {
        "name": name,
        "srcs": select({
            "@platforms//os:linux": ["linux.cc"],
            "@platforms//os:macos": ["macos.cc"],
            "//conditions:default": srcs,
        }),
        "copts": select({
            "@platforms//cpu:x86_64": ["-m64"],
            "@platforms//cpu:aarch64": ["-march=armv8-a"],
            "//conditions:default": [],
        }),
    }

SelectInfo = provider(
    doc = "A provider that uses select.",
    fields = {
        "value": "A value that may use select",
        "type": "The type of select",
    },
)

def _select_rule_impl(ctx):
    """Implementation that uses select."""
    return [SelectInfo(
        value = ctx.attr.value,
        type = _SELECT_TYPE,
    )]

select_rule = rule(
    implementation = _select_rule_impl,
    doc = "A rule that uses select.",
    attrs = {
        "value": attr.string(
            doc = "A string value (can use select)",
            default = select({
                "//conditions:default": "default",
            }),
        ),
    },
)
