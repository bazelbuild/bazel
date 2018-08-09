package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "//tools/java/runfiles:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = ["//tools/java/runfiles:embedded_tools"],
    visibility = ["//tools:__pkg__"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/java/runfiles:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
