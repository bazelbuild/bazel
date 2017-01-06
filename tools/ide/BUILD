licenses(["notice"])  # Apache 2.0

filegroup(
    name = "intellij_info",
    srcs = [
        "intellij_info.bzl",
        "intellij_info_impl.bzl",
    ],
    visibility = [
        "//src/test/java/com/google/devtools/build/lib:__pkg__",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools_srcs",
    srcs = [
        "BUILD.tools",
        "intellij_info.bzl",
        "intellij_info_impl.bzl",
    ],
    visibility = ["//visibility:public"],
)
