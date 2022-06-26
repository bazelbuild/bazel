load("@bazel_skylib//:bzl_library.bzl", "bzl_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

bzl_library(
    name = "defs",
    srcs = glob(["*.bzl"]),
    visibility = ["//visibility:public"],
)
