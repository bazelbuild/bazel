load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = [
        "//src/test/java/com/google/devtools/build/android/desugar:__subpackages__",
        "//src/tools/android/java/com/google/devtools/build/android/desugar:__subpackages__",
    ],
)

licenses(["notice"])  # Apache 2.0

java_library(
    name = "preanalysis",
    srcs = glob(["*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android/desugar/io",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel",
        "//src/tools/android/java/com/google/devtools/build/android/desugar/langmodel:desugar_class_attr_java_proto",
        "//third_party:asm",
        "//third_party:asm-commons",
        "//third_party:asm-tree",
        "//third_party:auto_value",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/tools/android/java/com/google/devtools/build/android/desugar:__pkg__"],
)
