"""
Tests that the documentation is generated even when unknown modules, calls, or providers are used.
"""

def exercise_the_api():
    _var1 = my_module
    _var2 = StructureOrProvider
    _var3 = my_module.Structure
    _var4 = my_module.sub_module.Structure
    _var5 = call("", 1, True, ["list"], {"key": "map"})
    _var6 = my_module.call("", 1, True, ["list"], {"key": "map"})
    _var7 = my_module.sub_module.call("xxx", 1, True, ["list"], {"key": "map"})
    _var8 = native.my_native_rule(name = "foo")

exercise_the_api()

def my_rule_impl(ctx):
    return struct()

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This rule does my things.",
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, mandatory = False),
    },
)
