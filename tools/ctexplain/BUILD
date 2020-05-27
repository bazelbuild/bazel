py_binary(
    name = "ctexplain",
    srcs = ["ctexplain.py"],
    deps = [":bazel_api"],
    python_version = "PY3",
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
        "//src/test/py/bazel:test_base",
        ":bazel_api",
    ],
)

py_library(
    name = "base",
    srcs = [
        "types.py",
    ],
    srcs_version = "PY3ONLY",
)
