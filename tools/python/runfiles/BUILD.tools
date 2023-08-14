load("//tools/python:gen_runfiles_constants.bzl", "gen_runfiles_constants")
load("@rules_python//python:defs.bzl", "py_library")

py_library(
    name = "runfiles",
    srcs = [
        "runfiles.py",
        ":_runfiles_constants",
    ],
    visibility = ["//visibility:public"],
)

gen_runfiles_constants(
    name = "_runfiles_constants",
)
