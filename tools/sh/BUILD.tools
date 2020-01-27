exports_files(["sh_toolchain.bzl", "sh_configure.bzl"])

toolchain_type(
    name = "toolchain_type",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)
