package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [".*"],  # .swp files and such
    ),
    visibility = ["//src/tools/runfiles:__pkg__"],
)

filegroup(
    name = "java-srcs",
    srcs = [
        "Runfiles.java",
        "Util.java",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        ":java-srcs",
    ],
    visibility = ["//src/tools/runfiles:__pkg__"],
)

java_library(
    name = "runfiles",
    srcs = [":java-srcs"],
    visibility = ["//src/tools/runfiles:__pkg__"],
)

java_test(
    name = "RunfilesTest",
    srcs = ["RunfilesTest.java"],
    deps = [":test_deps"],
)

java_test(
    name = "UtilTest",
    srcs = ["UtilTest.java"],
    deps = [":test_deps"],
)

java_library(
    name = "test_deps",
    testonly = 1,
    exports = [
        ":mock-file",
        ":runfiles",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

java_library(
    name = "mock-file",
    testonly = 1,
    srcs = ["MockFile.java"],
    exports = ["//third_party:guava"],
    deps = ["//third_party:guava"],
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
    visibility = ["//src/tools/runfiles:__pkg__"],
)
