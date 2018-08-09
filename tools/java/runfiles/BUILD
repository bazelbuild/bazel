package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.tools",
    ],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = ["//tools/java:__pkg__"],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//tools/java:__pkg__"],
)
