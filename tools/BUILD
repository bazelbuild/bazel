package(default_visibility = ["//visibility:public"])

# Used by Bazel. If a test program depends on this target, it is
# assumed to be sharding-compliant.
exports_files(["test_sharding_compliant"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/aquery_differ:srcs",
        "//tools/bash:srcs",
        "//tools/buildstamp:srcs",
        "//tools/build_defs:srcs",
        "//tools/build_rules:srcs",
        "//tools/config:srcs",
        "//tools/coverage:srcs",
        "//tools/distributions:srcs",
        "//tools/java:srcs",
        "//tools/jdk:srcs",
        "//tools/launcher:srcs",
        "//tools/def_parser:srcs",
        "//tools/platforms:srcs",
        "//tools/genrule:srcs",
        "//tools/cpp:srcs",
        "//tools/j2objc:srcs",
        "//tools/objc:srcs",
        "//tools/osx:srcs",
        "//tools/osx/crosstool:srcs",
        "//tools/test:srcs",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:srcs",
        "//tools/test/CoverageOutputGenerator/javatests/com/google/devtools/coverageoutputgenerator:srcs",
        "//tools/python:srcs",
        "//tools/runfiles:srcs",
        "//tools/sh:srcs",
        "//tools/allowlists:srcs",
        "//tools/zip:srcs",
    ],
)

# Sources in the built in repo '@bazel_tools'
filegroup(
    name = "embedded_tools_srcs",
    srcs = glob(["**"]) + [
        "//tools/android:embedded_tools",
        "//tools/bash:embedded_tools",
        "//tools/build_defs:embedded_tools",
        "//tools/build_rules:embedded_tools_srcs",
        "//tools/buildstamp:srcs",
        "//tools/config:srcs",
        "//tools/coverage:srcs",
        "//tools/cpp:embedded_tools",
        "//tools/genrule:srcs",
        "//tools/java:embedded_tools",
        "//tools/j2objc:srcs",
        "//tools/jdk:package-srcs",
        "//tools/jdk:srcs",
        "//tools/launcher:srcs",
        "//tools/def_parser:srcs",
        "//tools/platforms:srcs",
        "//tools/objc:srcs",
        "//tools/python:embedded_tools",
        "//tools/runfiles:embedded_tools",
        "//tools/test:embedded_tools",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:embedded_tools",
        "//tools/osx/crosstool:srcs",
        "//tools/osx:srcs",
        "//tools/sh:embedded_tools",
        "//tools/allowlists:srcs",
        "//tools/zip:srcs",
    ],
)

# All bzl files in the built in repo '@bazel_tools'.
filegroup(
    name = "bzl_srcs",
    srcs = [
        "//tools/android:bzl_srcs",
        "//tools/build_defs:bzl_srcs",
        "//tools/build_rules:bzl_srcs",
        "//tools/config:common_settings.bzl",
        "//tools/cpp:bzl_srcs",
        "//tools/jdk:bzl_srcs",
        "//tools/osx:bzl_srcs",
        "//tools/python:bzl_srcs",
        "//tools/sh:bzl_srcs",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/android:all_windows_tests",
        "//tools/aquery_differ:aquery_differ_test",
        "//tools/aquery_differ:aquery_differ_v2_test",
        "//tools/bash:all_windows_tests",
        "//tools/build_defs:all_windows_tests",
        "//tools/cpp/runfiles:all_windows_tests",
        "//tools/java:all_windows_tests",
        "//tools/jdk:all_windows_tests",
        "//tools/python:all_windows_tests",
        "//tools/test:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)

sh_binary(
    name = "upload_bazel_mirror",
    srcs = ["upload_bazel_mirror.sh"],
)
