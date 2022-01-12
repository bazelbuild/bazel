# Needed for #9006. Consider replacing label with @rules_python as per #9029 if
# merging from upstream becomes an issue.
load("//tools/python:private/defs.bzl", "py_library")

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
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = ["//third_party/py/six"],
)
