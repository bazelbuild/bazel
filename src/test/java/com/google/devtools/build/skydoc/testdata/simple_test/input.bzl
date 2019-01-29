def my_rule_impl(ctx):
    return struct()

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "first": attr.label(
            mandatory = True,
            doc = "first doc string",
            allow_single_file = True,
        ),
        "second": attr.string_dict(mandatory = True),
        "third": attr.output(mandatory = True),
        "fourth": attr.bool(default = False, doc = "fourth doc string", mandatory = False),
        "_hidden": attr.string(),
    },
)
