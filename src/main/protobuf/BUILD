package(default_visibility = ["//visibility:public"])

load("//tools/build_rules:genproto.bzl", "cc_grpc_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")
load("//third_party/protobuf/3.2.0:protobuf.bzl", "cc_proto_library", "py_proto_library")
load("//third_party/grpc:build_defs.bzl", "java_grpc_library")

FILES = [
    "build",
    "deps",
    "java_compilation",
    "crosstool_config",
    "extra_actions_base",
    "intellij_ide_info",
    "package_manifest",
    "test_status",
    "plmerge",
    "bundlemerge",
    "xcodegen",
    "worker_protocol",
    "invocation_policy",
    "android_deploy_info",
    "apk_manifest",
    "command_server",
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
    ],
)
