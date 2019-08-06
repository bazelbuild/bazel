load("//third_party/bazel_rules/rules_java/java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_library(
    name = "common",
    srcs = glob(["*.java"]),
    tags = ["bazel"],
    deps = [
        "//third_party:guava",
    ],
)
