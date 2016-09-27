package(default_visibility = ["//src:__subpackages__"])

load("//tools/build_rules:genproto.bzl", "java_proto_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

java_proto_library(
    name = "build_event_stream_java_proto",
    src = "build_event_stream.proto",
)
