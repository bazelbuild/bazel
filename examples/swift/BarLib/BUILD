package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load("//tools/build_defs/apple:swift.bzl", "swift_library")

swift_library(
    name = "BarLib",
    srcs = ["mul.swift"],
    module_name = "examples_BarLib",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)
