load("@bazel_pip_dev_deps//:requirements.bzl", "requirement")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)

alias(
    name = "abseil",
    actual = requirement("absl-py"),
    visibility = ["//visibility:public"],
)
