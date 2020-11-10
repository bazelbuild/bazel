"""Stardoc test to verify some apple-related parts of the build API."""

def exercise_the_api():
    var1 = apple_common.platform_type.ios
    var2 = apple_common.AppleDynamicFramework
    var3 = apple_common.platform.ios_device

exercise_the_api()

def my_rule_impl(ctx):
    return []

apple_related_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does apple-related things.",
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
