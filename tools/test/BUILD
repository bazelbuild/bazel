package(default_visibility = ["//visibility:public"])

# Members of this filegroup shouldn't have duplicate basenames, otherwise
# TestRunnerAction#getRuntimeArtifact() will get confused.
# Deprecated, do not use.
filegroup(
    name = "runtime",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "test_setup",
    srcs = ["test-setup.sh"],
)

filegroup(
    name = "test_xml_generator",
    srcs = ["generate-xml.sh"],
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

cc_binary(
    name = "test_wrapper",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["windows/test_wrapper.cc"],

        # Dummy source file, so this rule can be built on non-Windows platforms
        # (when building all targets in this package).
        "//conditions:default": ["windows/dummy.cc"],
    }),
    visibility = ["//visibility:private"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "test-setup.sh",
        "generate-xml.sh",
        "collect_coverage.sh",
    ] + glob(["LcovMerger/**"]) + select({
        "@bazel_tools//src/conditions:windows": ["test_wrapper"],
        "//conditions:default": [],
    }),
    visibility = ["//tools:__pkg__"],
)
