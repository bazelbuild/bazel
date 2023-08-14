load("@rules_java//java:defs.bzl", "java_proto_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library_srcs(
    name = "dist_jars",
    deps = [":build_event_stream_java_proto"],
)

java_proto_library(
    name = "build_event_stream_java_proto",
    # The Bazel IntelliJ plugin uses this.
    visibility = ["//visibility:public"],
    deps = ["build_event_stream_proto"],
)

proto_library(
    name = "build_event_stream_proto",
    srcs = ["build_event_stream.proto"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/packages/metrics:package_load_metrics_proto",
        "//src/main/protobuf:action_cache_proto",
        "//src/main/protobuf:command_line_proto",
        "//src/main/protobuf:failure_details_proto",
        "//src/main/protobuf:invocation_policy_proto",
        "@com_google_protobuf//:duration_proto",
        "@com_google_protobuf//:timestamp_proto",
    ],
)
