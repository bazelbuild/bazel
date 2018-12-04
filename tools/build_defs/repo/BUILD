load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

bzl_library(
    name = "lib",
    srcs = glob(["*.bzl"]),
)
