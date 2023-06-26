# Tools for gathering OSS licenses

load(":gather_packages.bzl", "packages_used")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//tools:__subpackages__",
        "@bazel_tools//tools:__subpackages__",
    ],
)

packages_used(
    name = "bazel_packages",
    target = "//src:bazel_nojdk",
    out = "bazel_packages.json",
)
