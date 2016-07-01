package(default_visibility = ["//visibility:public"])

# Used by Bazel. If a test program depends on this target, it is
# assumed to be sharding-compliant.
exports_files(["test_sharding_compliant"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/android/jack:srcs",
        "//tools/buildstamp:srcs",
        "//tools/build_defs/apple:srcs",
        "//tools/build_defs/apple/test:srcs",
        "//tools/build_defs/docker:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_rules:srcs",
        "//tools/jdk:srcs",
        "//tools/genrule:srcs",
        "//tools/cpp:srcs",
        "//tools/j2objc:srcs",
        "//tools/objc:srcs",
        "//tools/osx/crosstool:srcs",
        "//tools/test:srcs",
        "//tools/python:srcs",
        "//tools/zip:srcs",
    ],
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = glob(["**"]) + [
        "//tools/android/jack:srcs",
        "//tools/android:srcs",
        "//tools/build_defs/apple:srcs",
        "//tools/build_defs/docker:srcs",
        "//tools/build_defs/pkg:srcs",
        "//tools/build_rules:embedded_tools_srcs",
        "//tools/buildstamp:srcs",
        "//tools/cpp:srcs",
        "//tools/genrule:srcs",
        "//tools/j2objc:srcs",
        "//tools/jdk:package-srcs",
        "//tools/jdk:srcs",
        "//tools/objc:srcs",
        "//tools/python:srcs",
        "//tools/test:srcs",
        "//tools/osx/crosstool:srcs",
        "//tools/zip:srcs",
    ],
)
