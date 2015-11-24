# Description:
#   Zip provides a library for reading and writing zip files, allowing more
#   advanced manipulation than the JDK equivalents by providing detailed zip
#   entry data and raw file access.
#   Open-sourced as part of Bazel.
package(default_visibility = ["//src:__subpackages__"])

# Necessary for the bootstrap compilation.
filegroup(
    name = "java-srcs",
    srcs = glob(["*.java"]),
)

java_library(
    name = "zip",
    srcs = glob(["*.java"]),
    deps = ["//third_party:jsr305"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
