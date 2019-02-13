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
    name = "collect_cc_coverage",
    srcs = ["collect_cc_coverage.sh"],
)

filegroup(
    name = "coverage_report_generator",
    srcs = ["@bazel_tools//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:Main"],
)

# Test wrapper binary to run tests on Windows.
# This target just wraps the actual code in "tw_lib" to make it into a binary.
# See https://github.com/bazelbuild/bazel/issues/5508
cc_binary(
    name = "tw",
    srcs = ["windows/tw_main.cc"],
    visibility = ["//visibility:private"],
    deps = [":tw_lib"],
)

cc_binary(
    name = "xml",
    srcs = ["windows/xml_main.cc"],
    visibility = ["//visibility:private"],
    deps = [":tw_lib"],
)

# Test wrapper binary to run tests on Windows.
# See https://github.com/bazelbuild/bazel/issues/5508
cc_library(
    name = "tw_lib",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["windows/tw.cc"],
        "//conditions:default": ["empty_test_wrapper.cc"],
    }),
    hdrs = select({
        "@bazel_tools//src/conditions:windows": ["windows/tw.h"],
        "//conditions:default": [],
    }),
    linkopts = select({
        "//src/conditions:windows": [
            "-DEFAULTLIB:advapi32.lib",  # GetUserNameW
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:private"],
    deps = select({
        "@bazel_tools//src/conditions:windows": [
            "//src/main/cpp/util:filesystem",
            "//src/main/cpp/util:strings",
            "//src/main/native/windows:lib-file",
            "//src/main/native/windows:lib-util",
            "//third_party/ijar:zip",
            "@bazel_tools//tools/cpp/runfiles",
        ],
        "//conditions:default": [],
    }),
)

cc_test(
    name = "tw_test",
    srcs = select({
        "@bazel_tools//src/conditions:windows": ["windows/tw_test.cc"],
        "//conditions:default": ["empty_test.cc"],
    }),
    deps = [
        ":tw_lib",
        "//src/main/cpp/util:filesystem",
        "//src/test/cpp/util:windows_test_util",
        "@com_google_googletest//:gtest_main",
    ] + select({
        "@bazel_tools//src/conditions:windows": [
            "//src/main/native/windows:lib-file",
            "//src/main/native/windows:lib-util",
            "//third_party/ijar:zip",
        ],
        "//conditions:default": [],
    }),
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
        "collect_cc_coverage.sh",
    ] + glob(["CoverageOutputGenerator/**"]) + select({
        "@bazel_tools//src/conditions:windows": [
            "tw",
            "xml",
        ],
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
        "//tools/test/CoverageOutputGenerator/javatests/com/google/devtools/coverageoutputgenerator:all_windows_tests",
    ],
    visibility = ["//tools:__pkg__"],
)
