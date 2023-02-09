py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    deps = [":lib"],
)

filegroup(
    name = "srcs",
    srcs = ["BUILD"] + glob(["**/*.py"]),
    visibility = ["//examples:__pkg__"],
)
