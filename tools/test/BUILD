package(default_visibility = ["//visibility:public"])

# Members of this filegroup shouldn't have duplicate basenames, otherwise
# TestRunnerAction#getRuntimeArtifact() will get confused.
filegroup(
    name = "runtime",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "coverage_support",
    srcs = ["//tools/coverage:coverage_support"],
)

filegroup(
    name = "coverage_report_generator",
    srcs = ["//tools/coverage:coverage_report_generator"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)
