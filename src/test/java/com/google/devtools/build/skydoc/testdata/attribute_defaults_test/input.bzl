"""A golden test to verify attribute default values."""

def _my_rule_impl(ctx):
    return []

def _my_aspect_impl(target, ctx):
    return []

my_aspect = aspect(
    implementation = _my_aspect_impl,
    doc = "This is my aspect. It does stuff.",
    attr_aspects = ["deps", "attr_aspect"],
    attrs = {
        "_x": attr.label(mandatory = True),
        "y": attr.string(default = "why", doc = "some string"),
        "z": attr.string(mandatory = True),
    },
)

my_rule = rule(
    implementation = _my_rule_impl,
    doc = "This is my rule. It does stuff.",
    attrs = {
        "a": attr.bool(default = False, doc = "Some bool"),
        "b": attr.int(default = 2, doc = "Some int"),
        "c": attr.int_list(default = [0, 1], doc = "Some int_list"),
        "d": attr.label(default = "//foo:bar", doc = "Some label"),
        "e": attr.label_keyed_string_dict(
            default = {"//foo:bar": "hello", "//bar:baz": "goodbye"},
            doc = "Some label_keyed_string_dict",
        ),
        "f": attr.label_list(default = ["//foo:bar", "//bar:baz"], doc = "Some label_list"),
        "g": attr.string(default = "", doc = "Some string"),
        "h": attr.string_dict(
            default = {"animal": "bunny", "color": "orange"},
            doc = "Some string_dict",
        ),
        "i": attr.string_list(default = ["cat", "dog"], doc = "Some string_list"),
        "j": attr.string_list_dict(
            default = {"animal": ["cat", "bunny"], "color": ["blue", "orange"]},
            doc = "Some string_list_dict",
        ),
        "k": attr.bool(mandatory = True, doc = "Some bool"),
        "l": attr.int(mandatory = True, doc = "Some int"),
        "m": attr.int_list(mandatory = True, doc = "Some int_list"),
        "n": attr.label(mandatory = True, doc = "Some label"),
        "o": attr.label_keyed_string_dict(mandatory = True, doc = "Some label_keyed_string_dict"),
        "p": attr.label_list(mandatory = True, doc = "Some label_list"),
        "q": attr.string(mandatory = True, doc = "Some string"),
        "r": attr.string_dict(mandatory = True, doc = "Some string_dict"),
        "s": attr.string_list(mandatory = True, doc = "Some string_list"),
        "t": attr.string_list_dict(mandatory = True, doc = "Some string_list_dict"),
        "u": attr.string(),
        "v": attr.label(),
        "w": attr.int()
    },
)
