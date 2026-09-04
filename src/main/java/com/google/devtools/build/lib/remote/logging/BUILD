load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_library(
    name = "logging",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/protobuf:remote_execution_log_java_proto",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:timestamp_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/flogger",
        "//third_party/java/grpc:context",
        "//third_party/java/grpc:core",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
