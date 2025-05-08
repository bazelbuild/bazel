load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "abseil",
    srcs = glob(["**/*.py"]),
    imports = ["."],
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
)
