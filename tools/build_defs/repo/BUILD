load("@bazel_skylib//:skylark_library.bzl", "skylark_library")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

skylark_library(
    name = "lib",
    srcs = glob(["*.bzl"]),
)
