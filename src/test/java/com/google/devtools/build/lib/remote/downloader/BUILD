load("@rules_java//java:defs.bzl", "java_library", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]),
)

java_library(
    name = "RemoteDownloaderTests_lib",
    srcs = glob(
        [
            "*.java",
        ],
    ),
    deps = [
        "//src/main/java/com/google/devtools/build/lib/authandtls",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/cache",
        "//src/main/java/com/google/devtools/build/lib/bazel/repository/downloader",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream",
        "//src/main/java/com/google/devtools/build/lib/buildeventstream/proto:build_event_stream_java_proto",
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/events",
        "//src/main/java/com/google/devtools/build/lib/remote:ReferenceCountedChannel",
        "//src/main/java/com/google/devtools/build/lib/remote:Retrier",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/downloader",
        "//src/main/java/com/google/devtools/build/lib/remote/options",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:futures",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/test/java/com/google/devtools/build/lib/remote/util",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//src/test/java/com/google/devtools/build/lib/testutil:TestUtils",
        "//third_party/google/rpc:code_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/auth_kit:credentials",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:stub",
        "//third_party/java/grpc:util",
        "//third_party/java/guava:collect",
        "//third_party/java/guava:concurrent",
        "//third_party/java/guava:io",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/protobuf/util:time",
        "//third_party/java/rxjava3",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_asset_v1_remote_asset_java_grpc",
        "@remoteapis//:build_bazel_remote_asset_v1_remote_asset_java_proto",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

java_test(
    name = "RemoteDownloaderTests",
    tags = [
        "requires-network",
        "rules",
    ],
    test_class = "com.google.devtools.build.lib.AllTests",
    runtime_deps = [
        ":RemoteDownloaderTests_lib",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
    ],
)
