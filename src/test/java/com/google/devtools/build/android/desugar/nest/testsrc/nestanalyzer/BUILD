load("@rules_java//java:defs.bzl", "java_library")

package(
    default_testonly = 1,
    default_visibility = ["//src/test/java/com/google/devtools/build/android/desugar:__subpackages__"],
)

licenses(["notice"])  # Apache 2.0

java_library(
    name = "analyzed_target",
    srcs = glob(["*.java"]),
    javacopts = select({
        "//src/tools/android/java/com/google/devtools/build/android/desugar/testing/junit:jdk11-config": [
            "-source 11",
            "-target 11",
        ],
        "//conditions:default": [],
    }),
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["*"]),
)
