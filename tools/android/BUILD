package(default_visibility = ["//tools:__pkg__"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
            ".*",
        ],
    ),
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "android_extensions.bzl",
    ],
)
