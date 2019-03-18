load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@io_bazel_skydoc//skylark:skylark.bzl", "skylark_doc")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.bzl"]) + ["BUILD.repo"],
    visibility = ["//tools/build_defs:__pkg__"],
)

bzl_library(
    name = "lib",
    srcs = glob(["*.bzl"]),
)

skylark_doc(
    name = "doc",
    srcs = [":lib"],
    format = "html",
    site_root = ".",
    strip_prefix = "tools/build_defs/repo/",
    visibility = ["//site:__subpackages__"],
)
