load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "signal",
    srcs = [
        "InterruptSignalHandler.java",
    ],
    deps = [
        "//third_party:guava",
    ],
)
