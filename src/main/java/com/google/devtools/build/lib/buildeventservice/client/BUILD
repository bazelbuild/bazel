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
    name = "command_context",
    srcs = ["CommandContext.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "lifecycle_event",
    srcs = ["LifecycleEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
    ],
)

java_library(
    name = "stream_event",
    srcs = ["StreamEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
    ],
)

java_library(
    name = "build_event_service_client",
    srcs = ["BuildEventServiceClient.java"],
    deps = [
        ":command_context",
        ":lifecycle_event",
        ":stream_event",
        "//src/main/java/com/google/devtools/build/lib/skybridge:skybridge_interface",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "build_event_service_grpc_client",
    srcs = ["BuildEventServiceGrpcClient.java"],
    runtime_deps = [
        # This is required for client TLS.
        "//third_party/java/netty_tcnative",
    ],
    deps = [
        ":build_event_service_client",
        ":build_event_service_proto_util",
        ":command_context",
        ":lifecycle_event",
        ":stream_event",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/skybridge:sc_only",
        "//third_party/google/devtools/build/v1:publish_build_event_java_proto",
        "//third_party/google/devtools/build/v1:publish_build_event_proto_grpc",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
    ],
)

java_library(
    name = "build_event_service_proto_util",
    srcs = ["BuildEventServiceProtoUtil.java"],
    deps = [
        ":command_context",
        ":lifecycle_event",
        ":stream_event",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//third_party/google/devtools/build/v1:build_events_java_proto",
        "//third_party/google/devtools/build/v1:build_status_java_proto",
        "//third_party/google/devtools/build/v1:publish_build_event_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/protobuf:timestamp_java_proto",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:collect",
        "//third_party/java/protobuf:protobuf_lite",
    ],
)
