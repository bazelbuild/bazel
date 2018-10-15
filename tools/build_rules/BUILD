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
        "filetest.sh",
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

sh_binary(
    name = "filetest",
    srcs = ["filetest.sh"],
    visibility = ["//visibility:public"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
