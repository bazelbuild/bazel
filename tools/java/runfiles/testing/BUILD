package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            ".*",
            "*~",
        ],  # .swp files and such
    ),
    visibility = ["//tools/java/runfiles:__pkg__"],
)

java_test(
    name = "RunfilesTest",
    srcs = ["RunfilesTest.java"],
    test_class = "com.google.devtools.build.runfiles.RunfilesTest",
    deps = [":test_deps"],
)

java_test(
    name = "UtilTest",
    srcs = ["UtilTest.java"],
    test_class = "com.google.devtools.build.runfiles.UtilTest",
    deps = [":test_deps"],
)

java_library(
    name = "test_deps",
    testonly = 1,
    exports = [
        ":mock-file",
        "//third_party:guava",
        "//third_party:guava-testlib",
        "//third_party:junit4",
        "//third_party:truth",
        "//tools/java/runfiles",
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
    visibility = ["//tools/java/runfiles:__pkg__"],
)
