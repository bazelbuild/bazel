load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/com/google/devtools/build/lib/analysis/starlark:__pkg__"],
)

java_test(
    name = "StarlarkConfigurationFieldProcessorTest",
    srcs = ["StarlarkConfigurationFieldProcessorTest.java"],
    resources = [":ProcessorTestFiles"],
    runtime_deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/config:fragment",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis/starlark/annotations/processor:annotation_preprocessor_lib",
        "//third_party/java/compile_testing",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)

filegroup(
    name = "ProcessorTestFiles",
    srcs = glob(["optiontestsources/*.java"]),
)
