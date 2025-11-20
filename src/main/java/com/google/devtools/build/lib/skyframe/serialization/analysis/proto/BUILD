load("@com_google_protobuf//bazel:java_proto_library.bzl", "java_proto_library")
load("@com_google_protobuf//bazel:proto_library.bzl", "proto_library")
load("//tools/build_rules:utilities.bzl", "java_library_srcs")

package(
    default_applicable_licenses = ["//:license"],
    default_visibility = ["//src:__subpackages__"],
)

proto_library(
    name = "analysis_cache_service_metadata_status_proto",
    srcs = ["analysis_cache_service_metadata_status.proto"],
)

java_proto_library(
    name = "analysis_cache_service_metadata_status_java_proto",
    deps = [":analysis_cache_service_metadata_status_proto"],
)

java_library_srcs(
    name = "dist_jars",
    deps = [":analysis_cache_service_metadata_status_java_proto"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__subpackages__"],
)
