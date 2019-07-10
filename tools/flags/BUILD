# Common scaler value flag types
# 
# These definitions are only for local use withing Bazel. Most users should
# use the flag types from github.com/bazelbuild/bazel-skylib.

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

exports_files(
    ["flags.bzl"],
    visibility = ["//visibility://tools:__subpackages__"],
)
