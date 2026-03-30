load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)

proto_library(
    name = "cgroups_info_proto",
    srcs = ["cgroups_info.proto"],
)

java_proto_library(
    name = "cgroups_info_java_proto",
    deps = [":cgroups_info_proto"],
)

java_library_srcs(
    name = "cgroups_info_java_proto_srcs",
    deps = [":cgroups_info_java_proto"],
)

filegroup(
    name = "dist_jars",
    srcs = [
        ":cgroups_info_java_proto_srcs",
    ],
)
