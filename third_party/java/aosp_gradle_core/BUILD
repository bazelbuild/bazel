load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(name = "embedded_aosp_gradle_core")

java_library(
    name = "aosp_gradle_core",
    srcs = glob(["java/**/*.java"]),
    deps = [
        "//src/tools/android/java/com/google/devtools/build/android:android_common_30_1_3",
        "//third_party:asm",
        "//third_party:guava",
    ],
)

alias(
    name = "aosp_gradle_core_30_1_3",
    actual = ":aosp_gradle_core",
)
