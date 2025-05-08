load("@rules_java//java:defs.bzl", "java_library")
load("//tools/build_rules:java_rules_skylark.bzl", "bootstrap_java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src/java_tools/buildjar:buildjar_package_group"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*.java"]) + ["BUILD"],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "jarhelper",
    srcs = [
        "JarCreator.java",
        "JarHelper.java",
    ],
)

#
## Bootstrapping using Starlark rules
#

bootstrap_java_library(
    name = "bootstrap_jarhelper",
    srcs = [
        "JarCreator.java",
        "JarHelper.java",
    ],
    tags = ["manual"],
    deps = [
        "//src/java_tools/buildjar/java/com/google/devtools/build/buildjar:starlark-deps",
    ],
)
