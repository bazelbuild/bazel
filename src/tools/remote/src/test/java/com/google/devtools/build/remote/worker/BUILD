load("@rules_java//java:defs.bzl", "java_library")

package(
    default_applicable_licenses = ["//:license"],
    default_testonly = 1,
    default_visibility = ["//src/tools/remote:__pkg__"],
)

java_library(
    name = "output_service",
    srcs = ["TestOutputService.java"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/remote:combined_cache",
        "//src/main/java/com/google/devtools/build/lib/remote/common",
        "//src/main/java/com/google/devtools/build/lib/remote/util:digest_util",
        "//src/main/java/com/google/devtools/build/lib/remote/util:futures",
        "//src/main/java/com/google/devtools/build/lib/util:string_encoding",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//src/main/java/com/google/devtools/build/lib/vfs:pathfragment",
        "//src/main/java/com/google/devtools/common/options",
        "//src/main/protobuf:bazel_output_service_java_grpc",
        "//src/main/protobuf:bazel_output_service_java_proto",
        "//src/main/protobuf:bazel_output_service_rev2_java_proto",
        "//src/tools/remote/src/main/java/com/google/devtools/build/remote/worker",
        "//third_party/java/grpc:core",
        "//third_party/java/grpc:netty",
        "//third_party/java/grpc:stub",
        "//third_party/java/protobuf",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)

filegroup(
    name = "srcs",
    testonly = 0,
    srcs = glob(["**"]) + [
        "//src/tools/remote/src/test/java/com/google/devtools/build/remote/worker/http:srcs",
    ],
    visibility = ["//src/tools/remote:__pkg__"],
)
