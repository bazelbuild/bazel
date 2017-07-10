package(default_visibility = ["//src:__subpackages__"])

load("//tools/build_rules:utilities.bzl", "java_library_srcs")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_library_srcs(
    name = "dist_jars",
    deps = [":build_event_stream_java_proto"],
)

java_proto_library(
    name = "build_event_stream_java_proto",
    deps = ["build_event_stream_proto"],
)

proto_library(
    name = "build_event_stream_proto",
    srcs = ["build_event_stream.proto"],
    deps = [
        "//src/main/protobuf:invocation_policy_proto",
    ],
)
