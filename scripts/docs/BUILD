load("//tools/python:private/defs.bzl", "py_library", "py_binary")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//scripts:__pkg__"],
)

py_library(
    name = "dot-converter",
    srcs = ["dot_converter.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "generate_dot_graphs",
    srcs = ["generate_dot_graphs.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//site:__pkg__"],
    deps = [":dot-converter"],
)

exports_files(["jekyll_build.sh.tpl"])
