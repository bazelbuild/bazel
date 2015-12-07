package(default_visibility = ["//visibility:private"])

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = [
        "testenv.sh",
        "//src/test/shell/bazel:test-deps",
    ],
)

sh_test(
    name = "runfiles_test",
    size = "medium",
    srcs = ["runfiles_test.sh"],
    data = [":test-deps"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
