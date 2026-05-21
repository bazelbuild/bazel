load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "client",
    srcs = glob(["*.java"]),
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/remote/util",
        "//third_party:auto_value",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@com_google_protobuf//:protobuf_java",
        "@googleapis//google/devtools/build/v1:build_java_grpc",
        "@googleapis//google/devtools/build/v1:build_java_proto",
    ],
)

java_library(
    name = "build_event_service_grpc_client",
    srcs = ["BuildEventServiceGrpcClient.java"],
    runtime_deps = [
        # This is required for client TLS.
        "//third_party:netty_tcnative",
    ],
    deps = [
        ":build_event_service_client",
        ":build_event_service_proto_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@googleapis//google/devtools/build/v1:build_java_grpc",
        "@googleapis//google/devtools/build/v1:build_java_proto",
    ],
)