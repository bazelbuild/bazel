licenses(["notice"])  # Apache 2.0

sh_binary(
    name = "gensrcjar",
    srcs = ["gensrcjar.sh"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = ["test_rules.bzl"],
    visibility = ["//visibility:public"],
)
