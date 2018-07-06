load(":dep.bzl", "my_rule_impl")

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "first": attr.label(mandatory = True, allow_files = True, single_file = True),
        "second": attr.string_dict(mandatory = True),
    },
)

other_rule = rule(
    implementation = my_rule_impl,
    doc = "This is another rule.",
    attrs = {
        "third": attr.label(mandatory = True, allow_files = True, single_file = True),
        "fourth": attr.string_dict(mandatory = True),
    },
)

yet_another_rule = rule(
    implementation = my_rule_impl,
    doc = "This is yet another rule",
    attrs = {
        "fifth": attr.label(mandatory = True, allow_files = True, single_file = True),
    },
)
