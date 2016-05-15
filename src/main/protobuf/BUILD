package(default_visibility = ["//visibility:public"])

load("//tools/build_rules:genproto.bzl", "java_proto_library", "cc_grpc_library")
load("//third_party/protobuf:protobuf.bzl", "py_proto_library")

FILES = [
    "build",
    "dash",
    "deps",
    "java_compilation",
    "crosstool_config",
    "extra_actions_base",
    "android_studio_ide_info",
    "package_manifest",
    "test_status",
    "plmerge",
    "bundlemerge",
    "xcodegen",
    "worker_protocol",
    "invocation_policy",
    "android_deploy_info",
    "apk_manifest",
]

[java_proto_library(
    name = s + "_java_proto",
    src = s + ".proto",
) for s in FILES]

java_proto_library(
    name = "command_server_java_proto",
    src = "command_server.proto",
    use_grpc_plugin = True,
)

cc_grpc_library(
    name = "command_server_cc_proto",
    src = "command_server.proto",
)

java_proto_library(
    name = "remote_protocol_java_proto",
    src = "remote_protocol.proto",
    use_grpc_plugin = True,
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
