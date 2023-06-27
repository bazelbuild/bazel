# Tools for gathering OSS licenses

load(":gather_packages.bzl", "packages_used")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//tools:__subpackages__",
        "@bazel_tools//tools:__subpackages__",
    ],
)

packages_used(
    name = "bazel_packages",
    out = "bazel_packages.json",
    target = "//src:bazel_nojdk",
)

py_test(
    name = "packages_used_test",
    size = "medium",
    srcs = ["packages_used_test.py"],
    data = [
        ":bazel_packages.json",
    ],
    python_version = "PY3",
    deps = [
    ],
)
