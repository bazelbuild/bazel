def my_rule_impl(ctx):
    return struct()

def exercise_the_api():
    var1 = config_common.FeatureFlagInfo
    var2 = platform_common.TemplateVariableInfo
    var3 = repository_rule(implementation = my_rule_impl)
    var4 = testing.ExecutionInfo({})

exercise_the_api()

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule exercises some of the build API.",
    attrs = {
        "first": attr.label(mandatory = True, allow_files = True, single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
