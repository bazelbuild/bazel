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
    name = "test_wrapper_bin",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["windows/test_wrapper.cc"],
        "//conditions:default": ["empty_test_wrapper.cc"],
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
        "@bazel_tools//src/conditions:windows": ["test_wrapper_bin"],
        "//conditions:default": [],
    }),
    visibility = ["//tools:__pkg__"],
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
        "//tools/test/LcovMerger/javatests/com/google/devtools/lcovmerger:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
