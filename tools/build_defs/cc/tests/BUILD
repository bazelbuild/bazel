load(":cc_import_test.bzl", "cc_import_test_suite")

cc_import_test_suite(
    name = "cc_import_tests",
)

filegroup(
    name = "cc_import_tests_files",
    testonly = 1,
    srcs = glob(["**"]),
    visibility = ["//src/test/shell/bazel:__pkg__"],
)
