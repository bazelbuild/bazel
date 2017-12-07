package(default_visibility = ["//src:__subpackages__"])

load("//tools/build_rules:utilities.bzl", "java_library_srcs")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library_srcs(
    name = "dist_jars",
    deps = [":skylark_debugging_java_proto"],
)

java_proto_library(
    name = "skylark_debugging_java_proto",
    deps = ["skylark_debugging_proto"],
)

proto_library(
    name = "skylark_debugging_proto",
    srcs = ["skylark_debugging.proto"],
)
