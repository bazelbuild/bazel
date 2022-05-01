load("@rules_java//java:defs.bzl", "java_library")

package(default_visibility = ["//src:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

java_library(
    name = "runner",
    srcs = glob(["*.java"]),
    deps = [
        "//third_party/protobuf:protobuf_java_util",
        "//src/main/protobuf:worker_protocol_java_proto",
        "//third_party:gson",
    ],
)
