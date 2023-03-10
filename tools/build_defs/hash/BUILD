load("@rules_python//python:defs.bzl", "py_binary")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

exports_files(
    ["README.md"],
    visibility = ["//site:__pkg__"],
)

py_binary(
    name = "sha256",
    srcs = ["sha256.py"],
    visibility = ["//visibility:public"],
)

exports_files(
    ["hash.bzl"],
    visibility = ["//visibility:public"],
)

sh_test(
    name = "sha256_test",
    size = "small",
    srcs = ["sha256_test.sh"],
    data = ["sha256"],
    deps = ["@bazel_tools//tools/bash/runfiles"],
)
