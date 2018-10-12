licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = [
        "BUILD.tools",
        "test_rules.bzl",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_rules_test",
    srcs = ["test_rules_test.py"],
    data = ["test_rules.bzl"],
    deps = ["//src/test/py/bazel:test_base"],
)
