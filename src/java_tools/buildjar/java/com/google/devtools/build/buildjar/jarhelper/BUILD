package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
)

java_library(
    name = "jarhelper",
    srcs = [
        "JarCreator.java",
        "JarHelper.java",
    ],
)

#
## Bootstrapping using Skylark rules
#

load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

bootstrap_java_library(
    name = "bootstrap_jarhelper",
    srcs = [
        "JarCreator.java",
        "JarHelper.java",
    ],
    tags = ["manual"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:skylark-deps",
    ],
)
