package(default_visibility = ["//visibility:public"])

load("//tools:build_rules/genproto.bzl", "proto_java_library")

proto_library(
    name = "proto_smoke_test",
    srcs = ["test.proto"],
)

# proto_java_library is a quick and dirty rule to help Bazel compile itself.
proto_java_library(
    name = "test_proto",
    src = "test.proto",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "test.proto",
    ],
)
