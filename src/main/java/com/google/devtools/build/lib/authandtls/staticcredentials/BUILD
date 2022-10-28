load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "staticcredentials",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:auth",
        "//third_party:guava",
    ],
)
