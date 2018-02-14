package(default_visibility = ["//src:__subpackages__"])

load("//tools/build_rules:utilities.bzl", "java_library_srcs")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library_srcs(
    name = "dist_jars",
    deps = [":rpc_log_entry_java_proto"],
)

java_proto_library(
    name = "rpc_log_entry_java_proto",
    deps = ["rpc_log_entry_proto"],
)

proto_library(
    name = "rpc_log_entry_proto",
    srcs = ["rpc_log_entry.proto"],
    deps = [
        "@googleapis//:google_devtools_remoteexecution_v1test_remote_execution_proto",
        "@googleapis//:google_longrunning_operations_proto",
        "@googleapis//:google_rpc_status_proto",
        "@googleapis//:google_watch_v1_proto",
    ],
)
