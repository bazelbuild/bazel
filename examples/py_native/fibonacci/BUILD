filegroup(
    name = "srcs",
    srcs = glob(["*.py"]) + [
        "BUILD",
    ],
    visibility = ["//examples/py_native:__pkg__"],
)

py_library(
    name = "fibonacci",
    srcs = ["fib.py"],
    imports = ["."],
    visibility = ["//examples/py_native:__pkg__"],
)
