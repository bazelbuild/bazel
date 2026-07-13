load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "build_event_service_availability_event",
    srcs = ["BuildEventServiceAvailabilityEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "prev_invocation_bes_upload_report_failed_event",
    srcs = ["PrevInvocationBesUploadReportFailedEvent.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/events",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_event_service_options",
    srcs = ["BuildEventServiceOptions.java"],
    deps = [
        "//src/main/java/com/google/devtools/common/options",
    ],
)

java_library(
    name = "build_event_service_uploader",
    srcs = ["BuildEventServiceUploader.java"],
    deps = [
        ":bazel_event_impl",
        ":build_enqueued_impl",
        ":build_event_service_availability_event",
        ":build_finished_impl",
        ":invocation_finished_impl",
        ":invocation_started_impl",
        ":stream_finished_impl",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:build_event_service_client",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:command_context",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:lifecycle_event",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:stream_event",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:large-buildevent-serialized-event",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/profiler:auto_profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:error_prone_annotations",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_event_service_module",
    srcs = ["BuildEventServiceModule.java"],
    deps = [
        ":build_event_service_options",
        ":build_event_service_transport",
        ":command_context_impl",
        ":prev_invocation_bes_upload_report_failed_event",
        "//src/main/java/com/google/devtools/build/lib/analysis:test/test_configuration",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bugreport",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:build_event_service_client",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:command_context",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream:build_event_service_upload_complete_event",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/transports",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/network:connectivity_status",
        "//src/main/java/com/google/devtools/build/lib/profiler:auto_profiler",
        "//src/main/java/com/google/devtools/build/lib/profiler:google-auto-profiler-utils",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/build/lib/runtime:build_event_streamer",
        "//src/main/java/com/google/devtools/build/lib/runtime:common_command_options",
        "//src/main/java/com/google/devtools/build/lib/runtime:counting_artifact_group_namer",
        "//src/main/java/com/google/devtools/build/lib/runtime:instrumentation_output",
        "//src/main/java/com/google/devtools/build/lib/runtime:synchronized_output_stream",
        "//src/main/java/com/google/devtools/build/lib/runtime:target_summary_publisher",
        "//src/main/java/com/google/devtools/build/lib/runtime:ui_options",
        "//src/main/java/com/google/devtools/build/lib/util:abrupt_exit_exception",
        "//src/main/java/com/google/devtools/build/lib/util:detailed_exit_code",
        "//src/main/java/com/google/devtools/build/lib/util:exit_code",
        "//src/main/java/com/google/devtools/build/lib/util/io:out-err",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:flogger",
        "//third_party:guava",
        "//third_party:jsr305",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
    ],
)

java_library(
    name = "bazel_build_event_service_module",
    srcs = ["BazelBuildEventServiceModule.java"],
    deps = [
        ":build_event_service_module",
        ":build_event_service_options",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_directories",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/authandtls:grpc",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper:credential_module",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:build_event_service_client",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:build_event_service_grpc_client",
        "//src/main/java/com/google/devtools/build/lib/runtime:blaze_command_cluster",
        "//src/main/java/com/google/devtools/common/options",
        "//third_party:auth",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@com_google_protobuf//:protobuf_java",
    ],
)

java_library(
    name = "build_event_service_transport",
    srcs = ["BuildEventServiceTransport.java"],
    deps = [
        ":build_event_service_options",
        ":build_event_service_uploader",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:build_event_service_client",
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:command_context",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/util",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

IMPL_VISIBILITY = [
    "//src/main/java/com/google/devtools/build/lib/buildeventservice:__pkg__",
    "//src/test/java/com/google/devtools/build/lib/buildeventservice:__pkg__",
    "//java/com/google/devtools/build/lib/blaze/google/buildeventservice:__subpackages__",
    "//javatests/com/google/devtools/build/lib/blaze/google/buildeventservice:__subpackages__",
]

java_library(
    name = "command_context_impl",
    srcs = ["CommandContextImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:command_context",
        "//third_party:error_prone_annotations",
        "//third_party:guava",
        "//third_party:jsr305",
    ],
)

java_library(
    name = "build_enqueued_impl",
    srcs = ["BuildEnqueuedImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:lifecycle_event",
        "//third_party:guava",
    ],
)

java_library(
    name = "invocation_started_impl",
    srcs = ["InvocationStartedImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:lifecycle_event",
        "//third_party:guava",
    ],
)

java_library(
    name = "invocation_finished_impl",
    srcs = ["InvocationFinishedImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:lifecycle_event",
        "//third_party:guava",
    ],
)

java_library(
    name = "build_finished_impl",
    srcs = ["BuildFinishedImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:lifecycle_event",
        "//third_party:guava",
    ],
)

java_library(
    name = "bazel_event_impl",
    srcs = ["BazelEventImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:stream_event",
        "//third_party:guava",
    ],
)

java_library(
    name = "stream_finished_impl",
    srcs = ["StreamFinishedImpl.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/buildeventservice/client:stream_event",
        "//third_party:guava",
    ],
)
