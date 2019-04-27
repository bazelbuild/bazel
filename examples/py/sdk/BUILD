package(default_visibility = ["//visibility:public"])

py_binary(
    name = "main",
    srcs = ["main.py"],
    python_version = "PY3"
)

py_binary(
    name = "main_py2",
    main = "main.py",
    srcs = ["main.py"],
    python_version = "PY2"
)

load("@bazel_tools//tools/python:toolchain.bzl", "py_runtime_pair")
py_runtime_pair(
    name = "downloaded_sdk_runtimes",
    py2_runtime = "@python2.7//:runtime",
    py3_runtime = "@python3.7//:runtime",
)

toolchain(
    name = "downloaded_sdk_toolchain",
    toolchain = ":downloaded_sdk_runtimes",
    toolchain_type = "@bazel_tools//tools/python:toolchain_type"
)

filegroup(
    name = "srcs",
    srcs = [
        "main.py",
        "BUILD",
        "WORKSPACE",
    ]
)
