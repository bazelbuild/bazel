package(
    default_visibility = ["//src:__subpackages__"],
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
        "//src/main/java/com/google/devtools/build/lib:auth_and_tls_options",
        "//src/main/java/com/google/devtools/build/lib:build-base",
        "//src/main/java/com/google/devtools/build/lib:buildeventstream",
        "//src/main/java/com/google/devtools/build/lib:concurrent",
        "//src/main/java/com/google/devtools/build/lib:events",
        "//src/main/java/com/google/devtools/build/lib:io",
        "//src/main/java/com/google/devtools/build/lib:packages-internal",
        "//src/main/java/com/google/devtools/build/lib:runtime",
        "//src/main/java/com/google/devtools/build/lib:util",
        "//src/main/java/com/google/devtools/build/lib:vfs",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/exec/local",
        "//src/main/java/com/google/devtools/build/lib/standalone",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:option_filters_java_proto",
        "//third_party:apache_httpclient",
        "//third_party:apache_httpcore",
        "//third_party:auth",
        "//third_party:gson",
        "//third_party:guava",
        "//third_party:hazelcast",
        "//third_party:jsr305",
        "//third_party:netty",
        "//third_party/grpc:grpc-jar",
        "//third_party/protobuf:protobuf_java",
        "//third_party/protobuf:protobuf_java_util",
        "@googleapis//:google_bytestream_bytestream_java_grpc",
        "@googleapis//:google_bytestream_bytestream_java_proto",
        "@googleapis//:google_devtools_remoteexecution_v1test_remote_execution_java_grpc",
        "@googleapis//:google_devtools_remoteexecution_v1test_remote_execution_java_proto",
        "@googleapis//:google_longrunning_operations_java_proto",
        "@googleapis//:google_rpc_error_details_java_proto",
        "@googleapis//:google_rpc_status_java_proto",
        "@googleapis//:google_watch_v1_java_grpc",
        "@googleapis//:google_watch_v1_java_proto",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
