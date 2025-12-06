# Test file with deprecated parameter incompatible_use_toolchain_transition

def _impl(ctx):
    pass

# This uses a deprecated parameter that should be gracefully ignored
my_rule = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(),
    },
    incompatible_use_toolchain_transition = True,
)
