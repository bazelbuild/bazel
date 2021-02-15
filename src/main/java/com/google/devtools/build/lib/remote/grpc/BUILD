load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "grpc",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party:rxjava3",
        "//third_party/grpc:grpc-jar",
    ],
)
