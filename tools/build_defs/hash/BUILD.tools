load("@rules_python//python:py_binary.bzl", "py_binary")

licenses(["notice"])  # Apache 2.0

py_binary(
    name = "sha256",
    srcs = ["sha256.py"],
    visibility = ["//visibility:public"],
)

exports_files(
    ["hash.bzl"],
    visibility = ["//visibility:public"],
)
