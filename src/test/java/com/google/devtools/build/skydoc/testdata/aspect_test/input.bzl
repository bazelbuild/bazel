"""The input file for the aspect test"""

def my_aspect_impl(ctx):
    return []

my_aspect = aspect(
    implementation = my_aspect_impl,
    doc = "This is my aspect. It does stuff.",
    attr_aspects = ["deps", "attr_aspect"],
    attrs = {
        "first": attr.label(mandatory = True, allow_single_file = True),
        "second": attr.string_dict(mandatory = True),
    },
)

other_aspect = aspect(
    implementation = my_aspect_impl,
    doc = "This is another aspect.",
    attr_aspects = ["*"],
    attrs = {
        "_hidden": attr.string(),
        "third": attr.int(mandatory = True),
    },
)
