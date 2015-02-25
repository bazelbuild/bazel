package(default_visibility = ["//visibility:public"])

load("/tools/build_rules/py_rules", "py_library", "py_binary")

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    deps = [":lib"],
)
