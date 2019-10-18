filegroup(
    name = "srcs",
    srcs = glob(["*.py"]) + [
        "BUILD",
        "//examples/py_native/fibonacci:srcs",
    ],
    visibility = ["//examples:__pkg__"],
)

py_binary(
    name = "bin",
    srcs = ["bin.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lib",
        "//examples/py_native/fibonacci",
    ],
)

py_library(
    name = "lib",
    srcs = ["lib.py"],
    srcs_version = "PY3",
)

py_test(
    name = "test",
    srcs = ["test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":lib",
        "//examples/py_native/fibonacci",
    ],
)

py_test(
    name = "fail",
    srcs = ["fail.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [":lib"],
)
