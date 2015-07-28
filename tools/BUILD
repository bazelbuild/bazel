package(default_visibility = ["//visibility:public"])

# Used by Bazel. If a test program depends on this target, it is
# assumed to be sharding-compliant.
exports_files(["test_sharding_compliant"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/android/jack:srcs",
        "//tools/jdk:srcs",
        "//tools/genrule:srcs",
        "//tools/cpp:srcs",
        "//tools/objc:srcs",
        "//tools/test:srcs",
        "//tools/python:srcs",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]) + [
        "//tools/android:srcs",
        "//tools/android/jack:srcs",
        "//tools/jdk:package-srcs",
        "//tools/genrule:srcs",
        "//tools/cpp:srcs",
        "//tools/objc:srcs",
        "//tools/test:srcs",
        "//tools/python:srcs",
    ],
)
