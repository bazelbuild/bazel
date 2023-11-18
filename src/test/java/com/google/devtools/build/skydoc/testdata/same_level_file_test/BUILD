filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

exports_files(
    [
        "dep.bzl",
        "golden.textproto",
        "input.bzl",
    ],
)
