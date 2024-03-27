load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("//tools/python:gen_runfiles_constants.bzl", "gen_runfiles_constants")

package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/python:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "runfiles.py",
    ],
    visibility = ["//tools/python:__pkg__"],
)

py_library(
    name = "runfiles",
    testonly = 1,
    srcs = [
        "runfiles.py",
        ":_runfiles_constants",
    ],
)

gen_runfiles_constants(
    name = "_runfiles_constants",
)

py_test(
    name = "runfiles_test",
    srcs = ["runfiles_test.py"],
    visibility = ["//visibility:public"],
    deps = [":runfiles"],
)
