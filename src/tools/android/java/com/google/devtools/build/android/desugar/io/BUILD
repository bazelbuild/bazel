# Description:
#   Tool for desugaring Java constructs not supported by Android tools or devices.

load("@rules_java//java:defs.bzl", "java_library")

java_library(
    name = "io",
    srcs = glob(["*.java"]),
    visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/ziputils:ziputils_lib",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:jsr330_inject",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
