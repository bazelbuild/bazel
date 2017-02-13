package(default_visibility = ["//src/tools/benchmark:__pkg__"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

pkg_tar(
    name = "site",
    files = [
        "draw_chart.js",
        "line_chart.html",
    ],
)
