package(default_visibility = ["//visibility:public"])

load("//tools/build_rules:genproto.bzl", "cc_grpc_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")
load("//third_party/protobuf/3.4.0:protobuf.bzl", "py_proto_library")
load("//third_party/grpc:build_defs.bzl", "java_grpc_library")

exports_files(
    ["execution_statitics.proto"],
    visibility = [
        "//src/test/shell/integration:process_wrapper_test",
    ],
)

FILES = [
    "action_cache",
    "android_deploy_info",
    "bazel_flags",
    "build",
    "bundlemerge",
    "command_server",
    "crosstool_config",
    "deps",
    "desugar_deps",
    "execution_statistics",
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

proto_library(
    name = "command_line_proto",
    srcs = ["command_line.proto"],
    visibility = ["//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:__pkg__"],
    deps = [":option_filters_proto"],
)

java_proto_library(
    name = "command_line_java_proto",
    visibility = ["//src:__subpackages__"],
    deps = [":command_line_proto"],
)

java_library_srcs(
    name = "command_line_java_proto_srcs",
    visibility = ["//visibility:private"],
    deps = [":command_line_java_proto"],
)

cc_proto_library(
    name = "desugar_deps_cc_proto",
    deps = [":desugar_deps_proto"],
)

cc_proto_library(
    name = "worker_protocol_cc_proto",
    deps = [":worker_protocol_proto"],
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

java_library_srcs(
    name = "profile_java_proto_srcs",
    visibility = ["//visibility:private"],
    deps = ["//third_party/pprof:profile_java_proto"],
)

cc_proto_library(
    name = "execution_statistics_cc_proto",
    deps = [":execution_statistics_proto"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "dist_jars",
    srcs = [s + "_java_proto_srcs" for s in FILES] + [
        ":command_line_java_proto_srcs",
        ":command_server_java_grpc_srcs",
        ":option_filters_java_proto_srcs",
        ":profile_java_proto_srcs",
    ],
)
