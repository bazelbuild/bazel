package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/genproto", "proto_java_library")

FILES = [
    "build",
    "dash",
    "deps",
    "java_compilation",
    "crosstool_config",
    "extra_actions_base",
    "test_status",
    "bundlemerge",
    "xcodegen",
    "worker_protocol",
]

[proto_java_library(
    name = "proto_" + s,
    src = s + ".proto",
) for s in FILES]

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
