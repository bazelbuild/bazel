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
    name = "async_task_cache",
    srcs = ["AsyncTaskCache.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:rxjava3",
    ],
)

java_library(
    name = "rx_futures",
    srcs = ["RxFutures.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:rxjava3",
    ],
)

java_library(
    name = "rx_utils",
    srcs = ["RxUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party:rxjava3",
    ],
)

java_library(
    name = "utils",
    srcs = ["Utils.java"],
    deps = [
        ":digest_util",
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/actions:artifacts",
        "//src/main/java/com/google/devtools/build/lib/actions:execution_requirements",
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/authandtls/credentialhelper",
        "//src/main/java/com/google/devtools/build/lib/cmdline",
        "//src/main/java/com/google/devtools/build/lib/remote:ExecutionStatusException",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/common:bulk_transfer_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/common:cache_not_found_exception",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/protobuf:failure_details_java_proto",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@com_google_protobuf//:protobuf_java",
        "@com_google_protobuf//:protobuf_java_util",
        "@googleapis//google/rpc:rpc_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "tracing_metadata_utils",
    srcs = ["TracingMetadataUtils.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/analysis:blaze_version_info",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//third_party:guava",
        "//third_party:jsr305",
        "//third_party/grpc-java:grpc-jar",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "digest_util",
    srcs = ["DigestUtil.java"],
    deps = [
        ":digest_output_stream",
        "//src/main/java/com/google/devtools/build/lib/util:deterministic_writer",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/protobuf:spawn_java_proto",
        "//third_party:guava",
        "@com_google_protobuf//:protobuf_java",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_library(
    name = "digest_output_stream",
    srcs = ["DigestOutputStream.java"],
    deps = [
        "//third_party:guava",
        "//third_party:jsr305",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
