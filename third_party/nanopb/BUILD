licenses(["unencumbered"])

load("@rules_cc//cc:defs.bzl", "cc_library")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//third_party:__pkg__"],
)

cc_library(
    name = "nanopb",
    srcs = glob(["*.c"]),
    hdrs = glob(["*.h"]),
    includes = ["."],
    visibility = ["//visibility:public"],
)
