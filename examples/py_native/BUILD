filegroup(
    name = "srcs",
    srcs = glob(["*.py"]) + ["BUILD"],
    visibility = ["//examples:__pkg__"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    deps = [":lib"],
)

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_test(
    name = "test",
    srcs = ["test.py"],
    deps = [":lib"],
)

py_test(
    name = "fail",
    srcs = ["fail.py"],
    deps = [":lib"],
)
