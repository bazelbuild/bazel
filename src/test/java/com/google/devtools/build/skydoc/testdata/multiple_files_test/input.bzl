"""A direct dependency file of the input file."""

load(":testdata/multiple_files_test/dep.bzl", "my_rule_impl", "some_cool_function")

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "first": attr.label(
            mandatory = True,
            doc = "first my_rule doc string",
            allow_files = True,
            single_file = True,
        ),
        "second": attr.string_dict(mandatory = True),
    },
)

def top_fun(a, b, c):
    _ignore = [a, b, c]
    return 6

other_rule = rule(
    implementation = my_rule_impl,
    doc = "This is another rule.",
    attrs = {
        "third": attr.label(
            mandatory = True,
            doc = "third other_rule doc string",
            allow_files = True,
            single_file = True,
        ),
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
