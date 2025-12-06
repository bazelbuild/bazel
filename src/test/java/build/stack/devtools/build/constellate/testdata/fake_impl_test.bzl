# Test file where implementation comes from a load statement (simulated as FakeDeepStructure)

# Simulate loading an implementation function that becomes a FakeDeepStructure
_impl = some_module.some_function

# This should not fail even though _impl is a FakeDeepStructure, not a real function
my_rule = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(),
    },
)
