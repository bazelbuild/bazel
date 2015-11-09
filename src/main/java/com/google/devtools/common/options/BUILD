# Description:
#   A devtools-common options parser.
#   Open-sourced as part of Bazel.
package(default_visibility = ["//src:__subpackages__"])

java_library(
    name = "options",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
