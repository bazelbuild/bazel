load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/test/java/net/starlark/java:__pkg__"],
)

java_test(
    name = "StarlarkMethodProcessorTest",
    srcs = ["StarlarkMethodProcessorTest.java"],
    resources = glob(["testsources/*.java"]),
    runtime_deps = ["//src/main/java/net/starlark/java/eval"],  # loaded reflectively
    deps = [
        "//src/main/java/net/starlark/java/annot/processor:processor_lib",
        "//third_party/java/compile_testing",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/truth",
    ],
)
