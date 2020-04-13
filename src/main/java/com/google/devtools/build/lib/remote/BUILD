load("@rules_java//java:defs.bzl", "java_library")

package(
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]) + [
        "//src/main/java/com/google/devtools/build/lib/remote/common:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/downloader:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/disk:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/http:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/logging:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/merkletree:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/options:srcs",
        "//src/main/java/com/google/devtools/build/lib/remote/util:srcs",
    ],
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "remote",
    srcs = glob(
        ["*.java"],
        exclude = [
            "ExecutionStatusException.java",
            "ReferenceCountedChannel.java",
            "RemoteRetrier.java",
            "RemoteRetrierUtils.java",
            "Retrier.java",
        ],
    ),
    exports = [
        ":ExecutionStatusException",
        ":ReferenceCountedChannel",
        ":Retrier",
    ],
    deps = [
        ":ExecutionStatusException",
        ":ReferenceCountedChannel",
        ":Retrier",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform:platform_utils",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/collect",
        "//src/main/java/com/google/devtools/build/lib/collect/nestedset",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/packages",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/disk",
        "//src/main/java/com/google/devtools/build/lib/remote/downloader",
        "//src/main/java/com/google/devtools/build/lib/remote/http",
        "//src/main/java/com/google/devtools/build/lib/remote/logging",
        "//src/main/java/com/google/devtools/build/lib/remote/merkletree",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:output_service",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
        "@googleapis//:google_bytestream_bytestream_java_grpc",
        "@googleapis//:google_bytestream_bytestream_java_proto",
        "@googleapis//:google_longrunning_operations_java_proto",
        "@googleapis//:google_rpc_error_details_java_proto",
        "@googleapis//:google_rpc_status_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
        "@remoteapis//:build_bazel_semver_semver_java_proto",
    ],
)

java_library(
    name = "ExecutionStatusException",
    srcs = ["ExecutionStatusException.java"],
    deps = [
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
        "@googleapis//:google_rpc_status_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "ReferenceCountedChannel",
    srcs = ["ReferenceCountedChannel.java"],
    deps = [
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
    ],
)

java_library(
    name = "Retrier",
    srcs = [
        "RemoteRetrier.java",
        "RemoteRetrierUtils.java",
        "Retrier.java",
    ],
    deps = [
        ":ExecutionStatusException",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc:grpc-jar",
    ],
)
