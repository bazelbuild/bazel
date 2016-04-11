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

sh_test(
    name = "loading_phase_tests",
    size = "medium",
    srcs = ["loading_phase_tests.sh"],
    data = [":test-deps"],
    shard_count = 4,
)

sh_test(
    name = "experimental_ui_test",
    size = "medium",
    srcs = ["experimental_ui_test.sh"],
    data = [":test-deps"],
    shard_count = 4,
)

sh_test(
    name = "ui_test",
    size = "medium",
    srcs = ["ui_test.sh"],
    data = [":test-deps"],
)

test_suite(
    name = "all_tests",
    visibility = ["//visibility:public"],
)
