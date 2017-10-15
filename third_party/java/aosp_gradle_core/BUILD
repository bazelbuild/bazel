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
        "//third_party:android_common_25_0_0",
        "//third_party:asm",
        "//third_party:guava",
    ],
)
