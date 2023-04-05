load("@rules_python//python:defs.bzl", "py_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "futures",
    srcs = glob(["**/*.py"]),
    visibility = ["//visibility:public"],
)
