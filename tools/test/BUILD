package(default_visibility = ["//visibility:public"])

# Members of this filegroup shouldn't have duplicate basenames, otherwise
# TestRunnerAction#getRuntimeArtifact() will get confused.
filegroup(
    name = "runtime",
    srcs = [
        "test-setup.sh",
        "generate-xml.sh",
    ],
)

filegroup(
    name = "test_setup",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "collect_coverage",
    srcs = ["collect_coverage.sh"],
)

filegroup(
    name = "coverage_support",
    srcs = ["collect_coverage.sh"],
)

filegroup(
    name = "coverage_report_generator",
    srcs = ["@bazel_tools//tools/test/LcovMerger/java/com/google/devtools/lcovmerger:Main"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
