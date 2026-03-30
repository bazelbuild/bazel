load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    deps = [":lib"],
)

py_binary(
    name = "runfile",
    srcs = ["runfile.py"],
    data = ["//examples:runfile.txt"],
    deps = ["@rules_python//python/runfiles"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.py"]),
    visibility = ["//examples:__pkg__"],
)
