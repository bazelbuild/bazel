load("//tools/python:private/defs.bzl", "py_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "frozendict",
    srcs = glob(["**/*.py"]),
    srcs_version = "PY2AND3",
    imports = ["."],
    visibility = ["//visibility:public"],
)
