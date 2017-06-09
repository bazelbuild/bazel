package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

filegroup(
    name = "test-deps",
    testonly = 1,
    srcs = ["//src:bazel"],
)

py_library(
    name = "test_base",
    testonly = 1,
    srcs = ["test_base.py"],
    data = [":test-deps"],
)

py_test(
    name = "bazel_server_mode_test",
    size = "medium",
    srcs = ["bazel_server_mode_test.py"],
    deps = [":test_base"],
)
