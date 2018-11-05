filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/build_defs/apple:srcs",
        "//tools/build_defs/cc:srcs",
        "//tools/build_defs/hash:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_defs/repo:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/build_defs/apple:srcs",
        "//tools/build_defs/cc:srcs",
        "//tools/build_defs/hash:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_defs/repo:srcs",
    ],
    visibility = ["//tools:__pkg__"],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/build_defs/hash:all_windows_tests",
        "//tools/build_defs/pkg:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
