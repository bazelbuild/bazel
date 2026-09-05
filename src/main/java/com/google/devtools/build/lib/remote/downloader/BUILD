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
    name = "downloader",
    srcs = glob(["*.java"]),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/remote:ReferenceCountedChannel",
        "//src/main/java/com/google/devtools/build/lib/remote:Retrier",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/remote/util:futures",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/remote/util:utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party/google/rpc:code_java_proto",
        "//third_party/java/auth_kit:credentials",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:protobuf",
        "//third_party/java/guava:annotations",
        "//third_party/java/guava:base",
        "//third_party/java/protobuf/util:time",
        "@remoteapis//:build_bazel_remote_asset_v1_remote_asset_java_grpc",
        "@remoteapis//:build_bazel_remote_asset_v1_remote_asset_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
