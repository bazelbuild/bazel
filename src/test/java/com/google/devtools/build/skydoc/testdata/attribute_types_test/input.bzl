def my_rule_impl(ctx):
    return struct()

my_rule = rule(
    implementation = my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "a": attr.bool(mandatory = True, doc = "Some bool"),
        "b": attr.int(mandatory = True, doc = "Some int"),
        "c": attr.int_list(mandatory = True, doc = "Some int_list"),
        "d": attr.label(mandatory = True, doc = "Some label"),
        "e": attr.label_keyed_string_dict(mandatory = True, doc = "Some label_keyed_string_dict"),
        "f": attr.label_list(mandatory = True, doc = "Some label_list"),
        "g": attr.output(mandatory = False, doc = "Some output"),
        "h": attr.output_list(mandatory = False, doc = "Some output_list"),
        "i": attr.string(mandatory = True, doc = "Some string"),
        "j": attr.string_dict(mandatory = True, doc = "Some string_dict"),
        "k": attr.string_list(mandatory = True, doc = "Some string_list"),
        "l": attr.string_list_dict(mandatory = False, doc = "Some string_list_dict"),
    },
)
