# Needed for #9006. We can't easily replace this label with a @rules_python
# label (#9029) because this file is used in both @io_bazel and @bazel_tools,
# the latter of which can't reference other repos.
load("//tools/python:private/defs.bzl", "py_library")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "six",
    srcs = ["six.py"],
    imports = ["."],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)
