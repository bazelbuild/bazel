load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//visibility:public"])

java_library(
    name = "statistics",
    srcs = ["BlazeJavacStatistics.java"],
    deps = [
        "//third_party:auto_value",
        "//third_party:guava",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**/*.java"]) + [
        "BUILD",
    ],
    visibility = ["//src:__subpackages__"],
)
