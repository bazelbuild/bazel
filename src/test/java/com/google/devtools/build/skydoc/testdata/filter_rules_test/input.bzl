load(
    ":testdata/filter_rules_test/dep.bzl",
    "my_rule_impl",
    dep_rule = "my_rule",
)

def my_rule_impl(ctx):
    return struct()

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "first": attr.label(
            mandatory = True,
            doc = "first my_rule doc string",
            allow_single_file = True,
        ),
        "second": attr.string_dict(mandatory = True),
    },
)

other_rule = rule(
    implementation = my_rule_impl,
    doc = "This is another rule.",
    attrs = {
        "test": attr.string_dict(mandatory = True),
    },
)

whitelisted_dep_rule = dep_rule

yet_another_rule = rule(
    implementation = my_rule_impl,
    doc = "This is yet another rule",
    attrs = {
        "test": attr.string_dict(mandatory = True),
    },
)
