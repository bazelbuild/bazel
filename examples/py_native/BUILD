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
    legacy_create_init = False,
    deps = [
        ":lib",
        "//examples/py_native/fibonacci",
    ],
)

py_library(
    name = "lib",
    srcs = ["lib.py"],
)

py_test(
    name = "test",
    srcs = ["test.py"],
    legacy_create_init = False,
    deps = [
        ":lib",
        "//examples/py_native/fibonacci",
    ],
)

py_test(
    name = "fail",
    srcs = ["fail.py"],
    legacy_create_init = False,
    deps = [":lib"],
)
