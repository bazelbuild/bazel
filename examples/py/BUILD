package(default_visibility = ["//visibility:public"])

py_library(
    name = "lib",
    srcs = ["lib.py"],
    srcs_version = "PY3",
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":lib"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.py"]),
)
