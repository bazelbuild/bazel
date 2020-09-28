load("@rules_java//java:defs.bzl", "java_library")

licenses(["notice"])

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "annot",
    srcs = [
        "DocCategory.java",
        "DocumentMethods.java",
        "StarlarkConstructor.java",
    ],
)
