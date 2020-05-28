package(default_visibility = ["//visibility:public"])

py_binary(
    name = "ctexplain",
    srcs = ["ctexplain.py"],
    python_version = "PY3",
    deps = [":bazel_api"],
)

py_library(
    name = "bazel_api",
    srcs = ["bazel_api.py"],
    srcs_version = "PY3ONLY",
    deps = [":base"],
)

py_test(
    name = "bazel_api_test",
    size = "small",
    srcs = ["bazel_api_test.py"],
    deps = [
        ":bazel_api",
        "//src/test/py/bazel:test_base",
    ],
)

py_library(
    name = "base",
    srcs = [
        "types.py",
    ],
    srcs_version = "PY3ONLY",
)

py_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.py"],
    deps = [":base"]
)

filegroup(
    name = "srcs",
    srcs = glob(["*"]),
)