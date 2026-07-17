load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

licenses(["notice"])

py_library(
    name = "archive",
    srcs = ["archive.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "build_tar",
    srcs = ["build_tar.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools/build_defs:__pkg__"],
)
