package(
    default_visibility = [
        "//src/main/java/com/google/devtools/build/skydoc:__subpackages__",
        "//src/test/java/com/google/devtools/build/skydoc:__subpackages__",
    ],
)

load("//tools/build_rules:utilities.bzl", "java_library_srcs")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

proto_library(
    name = "stardoc_output_proto",
    srcs = ["stardoc_output.proto"],
)

java_proto_library(
    name = "stardoc_output_java_proto",
    deps = [":stardoc_output_proto"],
)

java_library_srcs(
    name = "dist_jars",
    visibility = ["//src:__pkg__"],
    deps = [":stardoc_output_java_proto"],
)
