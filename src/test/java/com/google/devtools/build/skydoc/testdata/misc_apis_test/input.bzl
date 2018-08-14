def my_rule_impl(ctx):
    return struct()

def exercise_the_api():
    var1 = config_common.FeatureFlagInfo
    var2 = platform_common.TemplateVariableInfo
    var3 = repository_rule(implementation = my_rule_impl)
    var4 = testing.ExecutionInfo({})
    # TODO(juliexxia): This isn't actually where this module would be used.
    # Move this to the rule definition when the relevant parameter is set up.
    var5 = config.int(flag = True)

exercise_the_api()

MyInfo = provider(
    fields = {
        "foo": "Something foo-related.",
        "bar": "Something bar-related.",
    },
)

my_info = MyInfo(foo="x", bar="y")

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule exercises some of the build API.",
    attrs = {
        "src": attr.label(
            doc = "The source file.",
            allow_files = [".bzl"]),
        "deps": attr.label_list(
            doc = """
A list of dependencies.
These dependencies better provide MyInfo!
...or else.
""",
            providers = [MyInfo],
            allow_files = False),
        "tool": attr.label(
            doc = "The location of the tool to use.",
            allow_files = True,
            default = Label("//foo/bar/baz:target",),
            cfg = "host",
            executable = True),
        "out": attr.output(
            doc = "The output file.",
            mandatory = True),
        "extra_arguments": attr.string_list(default = []),
    }
)
