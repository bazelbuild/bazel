load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//src:__pkg__"],
)

cc_binary(
    name = "simple_catter",
    srcs = ["simple_catter.cc"],
    visibility = ["//visibility:public"],
)
