# Bazel's Starlark interpreter

load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# The definitions of types.
java_library(
    name = "types",
    srcs = [
        "StarlarkType.java",
        "Types.java",
    ],
    visibility = ["//src/main/java/net/starlark/java:bazel"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
