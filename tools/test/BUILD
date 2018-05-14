package(default_visibility = ["//visibility:public"])

# Members of this filegroup shouldn't have duplicate basenames, otherwise
# TestRunnerAction#getRuntimeArtifact() will get confused.
filegroup(
    name = "runtime",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "test_setup",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "collect_coverage",
    srcs = ["collect_coverage.sh"],
)

java_binary(
    name = "LcovMerger",
    srcs = glob(["LcovMerger/java/**/*.java"]),
    main_class = "com.google.devtools.lcovmerger.Main",
)

java_library(
    name = "LcovMergerTestUtils",
    srcs = glob(["LcovMerger/java/**/*.java"]),
)

java_test(
    name = "LcovMergerTest",
    srcs = glob(["LcovMerger/javatests/**/*.java"]),
    deps = [
        ":LcovMergerTestUtils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/test/java/com/google/devtools/build/lib:foundations_testutil",
        "//src/test/java/com/google/devtools/build/lib:testutil",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "coverage_support",
    srcs = ["collect_coverage.sh"],
)

filegroup(
    name = "coverage_report_generator",
    srcs = ["//tools/coverage:coverage_report_generator"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
