load("@rules_python//python:py_binary.bzl", "py_binary")
load("@rules_python//python:py_library.bzl", "py_library")

py_library(
    name = "greeting",
    srcs = ["greeting.py"],
)

py_binary(
    name = "hello_world",
    srcs = ["hello_world.py"],
    deps = [
        ":greeting",
    ],
)
