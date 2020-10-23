load("//tools/python:private/defs.bzl", "py_test")

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
        "java_rules_skylark.bzl",
        "test_rules.bzl",
        "test_rules_private.bzl",
        "utilities.bzl",
    ],
    visibility = ["//visibility:public"],
)

py_test(
    name = "test_rules_test",
    srcs = ["test_rules_test.py"],
    data = [
        "test_rules.bzl",
        "test_rules_private.bzl",
    ],
    deps = ["//src/test/py/bazel:test_base"],
)
