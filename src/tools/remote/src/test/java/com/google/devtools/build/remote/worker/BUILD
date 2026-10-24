load("@rules_java//java:defs.bzl", "java_test")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/tools/remote:__pkg__"],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker/http:srcs",
    ],
    visibility = ["//src/tools/remote:__pkg__"],
)

java_test(
    name = "LiveOutputStreamsTest",
    srcs = ["LiveOutputStreamsTest.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/clock",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:tracing_metadata_utils",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs/inmemoryfs",
        "//src/main/java/com/google/devtools/common/options",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker",
        "//third_party/google/bytestream:bytestream_grpc",
        "//third_party/google/bytestream:bytestream_java_proto",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:inprocess",
        "//third_party/java/junit",
        "//third_party/java/protobuf",
        "//third_party/java/truth",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
