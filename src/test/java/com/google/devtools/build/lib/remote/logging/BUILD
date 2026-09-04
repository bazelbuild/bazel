load("@rules_java//java:defs.bzl", "java_test")

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

java_test(
    name = "logging",
    srcs = glob(["*.java"]),
    test_class = "com.google.devtools.build.lib.AllTests",
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote/logging",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/util/io",
        "//src/main/protobuf:remote_execution_log_java_proto",
        "//src/test/java/com/google/devtools/build/lib:test_runner",
        "//src/test/java/com/google/devtools/build/lib/testutil",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/google/longrunning:operations_java_proto",
        "//third_party/google/protobuf:any_java_proto",
        "//third_party/google/protobuf:timestamp_java_proto",
        "//third_party/google/rpc:status_java_proto",
        "//third_party/java/grpc:context",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/grpc:stub",
        "//third_party/java/grpc:util",
        "//third_party/java/guava:collect",
        "//third_party/java/junit",
        "//third_party/java/mockito",
        "//third_party/java/protobuf:protobuf_lite",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_grpc",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
