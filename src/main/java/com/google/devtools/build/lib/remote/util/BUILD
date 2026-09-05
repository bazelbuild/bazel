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
    name = "async_task_cache",
    srcs = ["AsyncTaskCache.java"],
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/rxjava3",
    ],
)

java_library(
    name = "rx_futures",
    srcs = ["RxFutures.java"],
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/rxjava3",
    ],
)

java_library(
    name = "rx_utils",
    srcs = ["RxUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/rxjava3",
    ],
)

java_library(
    name = "utils",
    srcs = ["Utils.java"],
    deps = [
        ":digest_util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:action_input",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/actions:file_metadata",
        "//src/main/java/com/google/devtools/build/lib/authandtls:grpc",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper",
        "//src/main/java/com/google/devtools/build/lib/remote:ExecutionStatusException",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/protobuf:duration_java_proto",
        "//third_party/google/protobuf:timestamp_java_proto",
        "//third_party/google/rpc:code_java_proto",
        "//third_party/google/rpc:error_details_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/guava:base",
        "//third_party/java/guava:concurrent",
        "//third_party/java/jsr305_annotations",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/protobuf/util:time",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "tracing_metadata_utils",
    srcs = ["TracingMetadataUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//third_party/java/grpc:context",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:protobuf",
        "//third_party/java/grpc:stub",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/jsr305_annotations",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "digest_util",
    srcs = ["DigestUtil.java"],
    deps = [
        ":digest_output_stream",
        "//src/main/java/com/google/devtools/build/lib/remote/common:action_key",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party/java/guava:base",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:hash",
        "//third_party/java/guava:io",
        "//third_party/java/protobuf",
        "//third_party/java/protobuf:protobuf_lite",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "digest_output_stream",
    srcs = ["DigestOutputStream.java"],
    deps = [
        "//third_party/java/guava:base",
        "//third_party/java/guava:hash",
        "//third_party/java/jsr305_annotations",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
