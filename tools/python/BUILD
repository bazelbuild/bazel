package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:srcs",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]) + [
        "private/py_test_alias.bzl",
    ],
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)
