load(":cc_import_test.bzl", "cc_import_test_suite")

licenses(["notice"])  # Apache 2.0

cc_import_test_suite(
    name = "cc_import_tests",
)

filegroup(
    name = "cc_import_tests_files",
    srcs = glob(["**"]),
    visibility = [
        "//src/test/shell/bazel:__pkg__",
        "//tools/build_defs:__pkg__",
        "//tools/build_defs/cc:__pkg__",
    ],
)
