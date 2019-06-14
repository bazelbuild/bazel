package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/bazel/repository:__pkg__"],
)

java_test(
    name = "DownloaderTestSuite",
    srcs = glob(["*.java"]),
    tags = [
        "rules",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:mockito",
        "//third_party:truth",
    ],
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
    tests = [
        ":windows_tests",
    ],
    visibility = ["//src/test/java/com/google/devtools/build/lib/bazel/repository:__pkg__"],
)
