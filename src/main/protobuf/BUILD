package(default_visibility = ["//visibility:public"])

load("//tools/build_rules:genproto.bzl", "cc_grpc_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")
load("//third_party/protobuf/3.2.0:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("//third_party/grpc:build_defs.bzl", "java_grpc_library")

FILES = [
    "android_deploy_info",
    "apk_manifest",
    "build",
    "bundlemerge",
    "command_server",
    "crosstool_config",
    "deps",
    "extra_actions_base",
    "invocation_policy",
    "java_compilation",
    "plmerge",
    "test_status",
    "worker_protocol",
]

[proto_library(
    name = s + "_proto",
    srcs = [s + ".proto"],
) for s in FILES]

[java_proto_library(
    name = s + "_java_proto",
    deps = [":" + s + "_proto"],
) for s in FILES]

[java_library_srcs(
    name = s + "_java_proto_srcs",
    deps = [":" + s + "_java_proto"],
) for s in FILES]

# This new option tagging method is in flux while being applied to the options
# in the Bazel code base. The visibility should not be changed to allow external
# dependencies until the interface has stabilized and can commit to maintaining
# backwards compatibility for 6 months' time.
# TODO(bazel-team) Make these visibility:public when the interface is stable.
proto_library(
    name = "option_filters_proto",
    srcs = ["option_filters.proto"],
    visibility = ["//visibility:private"],
)

java_proto_library(
    name = "option_filters_java_proto",
    visibility = ["//src:__subpackages__"],
    deps = [":option_filters_proto"],
)

java_library_srcs(
    name = "option_filters_java_proto_srcs",
    visibility = ["//visibility:private"],
    deps = [":option_filters_java_proto"],
)

cc_proto_library(
    name = "worker_protocol_cc_proto",
    srcs = ["worker_protocol.proto"],
    include = ".",
    default_runtime = "//third_party/protobuf:protobuf",
    protoc = "//third_party/protobuf:protoc",
)

java_grpc_library(
    name = "command_server_java_grpc",
    srcs = [":command_server_proto"],
    deps = [":command_server_java_proto"],
)

cc_grpc_library(
    name = "command_server_cc_proto",
    src = "command_server.proto",
)

py_proto_library(
    name = "build_pb_py",
    srcs = ["build.proto"],
    default_runtime = "//third_party/protobuf:protobuf_python",
    protoc = "//third_party/protobuf:protoc",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "dist_jars",
    srcs = [s + "_java_proto_srcs" for s in FILES] + [
        ":command_server_java_grpc_srcs",
        ":option_filters_java_proto_srcs",
    ],
)
