package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/genproto", "proto_java_library")

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
