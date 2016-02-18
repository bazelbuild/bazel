package(default_visibility = ["//visibility:public"])

load("//tools/build_rules:genproto.bzl", "java_proto_library")

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
    "remote_protocol",
]

[java_proto_library(
    name = s + "_proto",
    src = s + ".proto",
) for s in FILES]

java_proto_library(
    name = "build_proto_v2",
    src = "build.proto",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
