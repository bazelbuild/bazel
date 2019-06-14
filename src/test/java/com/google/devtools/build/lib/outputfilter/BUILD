package(
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib:__pkg__"],
)

java_test(
    name = "OutputFilterTests",
    srcs = glob(["*.java"]),
    # TODO(bazel-team) The test is currently broken due to BazelAnalysisMock not supporting integration tests.
    tags = ["manual"],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = ["//src/test/java/com/google/devtools/build/lib:test_runner"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:unix",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/outputfilter",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib:analysis_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//src/test/java/com/google/devtools/build/lib/buildtool:testutil",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
