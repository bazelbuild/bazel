load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

# Library of concurrency utilities.
java_library(
    name = "concurrent",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)
