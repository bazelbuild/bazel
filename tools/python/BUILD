package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/python/runfiles:embedded_tools",
    ],
    visibility = ["//tools:__pkg__"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/python/runfiles:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
