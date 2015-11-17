package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/build_rules/go/tools:srcs",
        "//tools/build_rules/go/toolchain:srcs",
    ],
    visibility = [
        "//src:__subpackages__",
        "//tools:__pkg__",
    ],
)
