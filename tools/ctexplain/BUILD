# Description:
#   Tool for measuring how configuration transitions affect build graph size.
load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

py_binary(
    name = "ctexplain",
    srcs = ["ctexplain.py"],
    python_version = "PY3",
    deps = [
        ":analyses",
        ":base",
        ":bazel_api",
        ":lib",
        "//third_party/py/abseil",
    ],
)

py_library(
    name = "lib",
    srcs = ["lib.py"],
    srcs_version = "PY3ONLY",
    deps = [
        ":base",
        ":bazel_api",
    ],
)

py_library(
    name = "bazel_api",
    srcs = ["bazel_api.py"],
    srcs_version = "PY3ONLY",
    deps = [":base"],
)

py_library(
    name = "analyses",
    srcs = ["analyses/summary.py"],
    srcs_version = "PY3ONLY",
    deps = [":base"],
)

py_library(
    name = "base",
    srcs = [
        "types.py",
        "util.py",
    ],
    srcs_version = "PY3ONLY",
    deps = [
        # Backport for Python < 3.7.
        "//third_party/py/frozendict",
    ],
)

py_test(
    name = "lib_test",
    size = "small",
    srcs = ["lib_test.py"],
    python_version = "PY3",
    deps = [
        ":bazel_api",
        ":lib",
        "//src/test/py/bazel:test_base",
    ],
)

py_test(
    name = "bazel_api_test",
    size = "small",
    srcs = ["bazel_api_test.py"],
    python_version = "PY3",
    deps = [
        ":bazel_api",
        "//src/test/py/bazel:test_base",
    ],
)

py_test(
    name = "analyses_test",
    size = "small",
    srcs = ["analyses/summary_test.py"],
    main = "analyses/summary_test.py",  # TODO: generalize this.
    python_version = "PY3",
    deps = [
        ":analyses",
        ":base",
    ],
)

py_test(
    name = "types_test",
    size = "small",
    srcs = ["types_test.py"],
    python_version = "PY3",
    deps = [
        ":base",
        "//third_party/py/frozendict",
    ],
)
