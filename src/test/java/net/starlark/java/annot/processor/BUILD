load("@rules_java//java:defs.bzl", "java_test")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "StarlarkMethodProcessorTest",
    srcs = ["StarlarkMethodProcessorTest.java"],
    resources = glob(["testsources/*.java"]),
    runtime_deps = ["//src/main/java/net/starlark/java/eval"],  # loaded reflectively
    deps = [
        "//src/main/java/net/starlark/java/annot/processor",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
)

test_suite(
    name = "all_windows_tests",
    tests = [
        ":windows_tests",
    ],
    visibility = ["//visibility:private"],
)
