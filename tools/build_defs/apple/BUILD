package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["*.bzl"]) + [
        "BUILD",
        "//tools/build_defs/apple/test:srcs",
    ],
    visibility = ["//tools/build_defs:__pkg__"],
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
    tests = [":windows_tests"],
    visibility = ["//tools/build_defs:__pkg__"],
)
