def exercise_the_api():
    var1 = java_common.JavaRuntimeInfo
    var2 = JavaInfo
    var3 = java_proto_common

exercise_the_api()

def my_rule_impl(ctx):
    return struct()

java_related_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does java-related things.",
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
