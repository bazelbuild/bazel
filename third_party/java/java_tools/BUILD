package(default_visibility = ["//visibility:public"])

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

filegroup(
    name = "java_tools",
    srcs = glob(
        ["*.jar"],
        exclude = ["bazel-singlejar_deploy.jar"],
    ) + select({
        "//src/conditions:arm": ["bazel-singlejar_deploy.jar"],
        "//conditions:default": [],
    }),
)

pkg_tar(
    name = "java_tools_pkg",
    srcs = glob(["*.jar"]) + ["BUILD.pkg"],
)
