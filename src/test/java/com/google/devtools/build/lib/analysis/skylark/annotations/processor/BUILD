load("@rules_java//java:defs.bzl", "java_test")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_test(
    name = "StarlarkConfigurationFieldProcessorTest",
    srcs = ["StarlarkConfigurationFieldProcessorTest.java"],
    resources = [":ProcessorTestFiles"],
    runtime_deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:config/fragment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/skylark/annotations/processor:annotation_preprocessor",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)

filegroup(
    name = "ProcessorTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
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
)
