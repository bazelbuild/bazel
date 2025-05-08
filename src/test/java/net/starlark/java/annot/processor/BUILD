load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

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
        "//src/main/java/net/starlark/java/annot/processor:processor_lib",
        "//third_party:compile_testing",
        "//third_party:guava",
        "//third_party:junit4",
        "//third_party:truth",
    ],
)
