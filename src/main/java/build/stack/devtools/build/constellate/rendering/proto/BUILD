load("@rules_java//java:defs.bzl", "java_proto_library")
load("@io_bazel//third_party/grpc:build_defs.bzl", "java_grpc_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")
load("@rules_proto//proto:defs.bzl", "proto_library")

package(
    default_visibility = [
        "//src/main/java/build/stack/devtools/build/constellate:__subpackages__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

proto_library(
    name = "starlark_language_proto",
    srcs = ["starlark_language.proto"],
)

java_proto_library(
    name = "starlark_language_java_proto",
    deps = [":starlark_language_proto"],
)

java_library_srcs(
    name = "dist_jars",
    visibility = ["//src:__pkg__"],
    deps = [":starlark_language_java_proto"],
)

proto_library(
    name = "starlark_server_proto",
    srcs = ["starlark_server.proto"],
    deps = [":starlark_language_proto"],
)

java_proto_library(
    name = "starlark_server_java_proto",
    deps = [":starlark_server_proto"],
)

java_grpc_library(
    name = "starlark_server_java_grpc",
    srcs = [":starlark_server_proto"],
    deps = [
        ":starlark_language_java_proto",
        ":starlark_server_java_proto",
    ],
)
