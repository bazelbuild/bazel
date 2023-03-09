# TODO(plf): describe this package.
cc_library(
    name = "foo",
    #srcs = ["foo.cc"],
    #hdrs = ["foo.h"],
    deps = [
        "baz",
    ],
)
cc_library(
    name = "baz",
    srcs = ["baz.cc"],
    hdrs = ["baz.h"],
    deps = [
        "bar"
    ],
)

cc_library(
    name = "bar",
    srcs = ["bar.cc"],
    hdrs = ["bar.h"],
    deps = [
    ],
)

cc_shared_library(
    name = "foo_shared",
    deps = [
        "foo",
        "bar"
    ],
)
