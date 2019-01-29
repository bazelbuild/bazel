package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            ".*",
            "*~",
        ],  # .swp files and such
    ) + ["//tools/java/runfiles/testing:srcs"],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":java-srcs",
    ],
    visibility = ["//tools/java:__pkg__"],
)

filegroup(
    name = "java-srcs",
    srcs = [
        "Runfiles.java",
        "Util.java",
    ],
)

java_library(
    name = "runfiles",
    srcs = [":java-srcs"],
    visibility = ["//tools/java/runfiles/testing:__pkg__"],
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
        "//tools/java/runfiles/testing:all_windows_tests",
    ],
    visibility = ["//tools/java:__pkg__"],
)
