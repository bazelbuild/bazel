load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])  # Apache 2.0

exports_files(glob(["*.bzl"]))

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs/repo:__pkg__"],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools/build_defs/repo:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.bzl"]) + ["BUILD.repo"],
    visibility = ["//tools/build_defs/repo:__pkg__"],
)

bzl_library(
    name = "lib",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools/build_defs/repo:__pkg__"],
)
