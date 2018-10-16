package(default_visibility = ["//visibility:public"])

# Used by Bazel. If a test program depends on this target, it is
# assumed to be sharding-compliant.
exports_files(["test_sharding_compliant"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/bash:srcs",
        "//tools/buildstamp:srcs",
        "//tools/build_defs:srcs",
        "//tools/build_rules:srcs",
        "//tools/coverage:srcs",
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
        "//tools/whitelists:srcs",
        "//tools/zip:srcs",
    ],
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = glob(["**"]) + [
        "//tools/android:embedded_tools",
        "//tools/bash:embedded_tools",
        "//tools/build_defs:embedded_tools",
        "//tools/build_rules:embedded_tools_srcs",
        "//tools/buildstamp:srcs",
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
        "//tools/whitelists:srcs",
        "//tools/zip:srcs",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        "//tools/android:all_windows_tests",
        "//tools/bash:all_windows_tests",
        "//tools/build_defs:all_windows_tests",
        "//tools/build_rules:all_windows_tests",
        "//tools/cpp/runfiles:all_windows_tests",
        "//tools/java:all_windows_tests",
        "//tools/jdk:all_windows_tests",
        "//tools/python:all_windows_tests",
        "//tools/test:all_windows_tests",
    ],
    visibility = ["//src:__pkg__"],
)
