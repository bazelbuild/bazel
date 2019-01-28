package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib:__pkg__"],
)

java_library(
    name = "remote",
    srcs = glob(["*.java"]),
    tags = ["bazel"],
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis/platform",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/concurrent",
        "//src/main/java/com/google/devtools/build/lib/profiler",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore",
        "//src/main/java/com/google/devtools/build/lib/remote/blobstore/http",
        "//src/main/java/com/google/devtools/build/lib/remote/logging",
        "//src/main/java/com/google/devtools/build/lib/remote/merkletree",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auth",
        "//third_party:guava",
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
