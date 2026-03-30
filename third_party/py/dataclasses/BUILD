# Description:
#   dataclasses module from the Python 3.7 standard library.
#   For backporting to Python 3.6 and below.
load("@rules_python//python:py_library.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Python Software Foundation

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

py_library(
    name = "dataclasses",
    srcs = glob(["**/*.py"]),
    imports = ["."],
    srcs_version = "PY3",  # Only works with Python 3.
)
