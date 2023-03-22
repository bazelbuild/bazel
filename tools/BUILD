package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/aquery_differ:srcs",
        "//tools/bash:srcs",
        "//tools/buildstamp:srcs",
        "//tools/build_defs:srcs",
        "//tools/build_rules:srcs",
        "//tools/compliance:srcs",
        "//tools/coverage:srcs",
        "//tools/ctexplain:srcs",
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
        "//tools/proto:srcs",
        "//tools/windows:srcs",
        "//tools/test:srcs",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:srcs",
        "//tools/test/CoverageOutputGenerator/javatests/com/google/devtools/coverageoutputgenerator:srcs",
        "//tools/python:srcs",
        "//tools/runfiles:srcs",
        "//tools/sh:srcs",
        "//tools/allowlists:srcs",
        "//tools/zip:srcs",
        "//tools/mini_tar:srcs",
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
        "//tools/coverage:srcs",
        "//tools/cpp:embedded_tools",
        "//tools/genrule:srcs",
        "//tools/java:embedded_tools",
        "//tools/java:java_stub_template.txt",
        "//tools/j2objc:srcs",
        "//tools/jdk:package-srcs",
        "//tools/jdk:srcs",
        "//tools/launcher:srcs",
        "//tools/def_parser:srcs",
        "//tools/windows:srcs",
        "//tools/platforms:srcs",
        "//tools/proto:srcs",
        "//tools/objc:srcs",
        "//tools/python:embedded_tools",
        "//tools/runfiles:embedded_tools",
        "//tools/test:embedded_tools",
        "//tools/test/CoverageOutputGenerator/java/com/google/devtools/coverageoutputgenerator:embedded_tools",
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
        "//tools/cpp:bzl_srcs",
        "//tools/jdk:bzl_srcs",
        "//tools/osx:bzl_srcs",
        "//tools/python:bzl_srcs",
        "//tools/sh:bzl_srcs",
        "//tools/test:bzl_srcs",
        "//tools/windows:bzl_srcs",
    ],
)

sh_binary(
    name = "upload_bazel_mirror",
    srcs = ["upload_bazel_mirror.sh"],
)
