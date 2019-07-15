# Common config settings and flags.
#
# These definitions are only for local use within Bazel built-in tools.
# In all other cases, use types from github.com/bazelbuild/bazel-skylib.

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "@bazel_tools//tools:__pkg__",
        "@bazel_tools//tools/build_defs:__pkg__",
    ],
)

exports_files(
    ["common_settings.bzl"],
    visibility = ["@bazel_tools//tools:__subpackages__"],
)
