package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src/main/java/com/google/devtools/build/lib/remote:__pkg__"],
)

java_library(
    name = "common",
    srcs = glob(["*.java"]),
    tags = ["bazel"],
    deps = [
        "//src/main/java/com/google/devtools/build/lib/actions",
        "//src/main/java/com/google/devtools/build/lib/remote/merkletree",
        "//src/main/java/com/google/devtools/build/lib/vfs",
        "//third_party:guava",
        "//third_party/protobuf/3.6.1:protobuf_java",
        "@remoteapis//:build_bazel_remote_execution_v2_remote_execution_java_proto",
    ],
)
