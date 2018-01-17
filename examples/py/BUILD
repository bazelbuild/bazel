package(default_visibility = ["//visibility:public"])

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    deps = [":lib"],
    legacy_create_init = False,
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.py"]),
)
