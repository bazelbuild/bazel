load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("@io_bazel_skydoc//stardoc:stardoc.bzl", "stardoc")
load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["*.bzl"]) + ["BUILD.repo"],
    visibility = ["//tools/build_defs:__pkg__"],
)

bzl_library(
    name = "lib",
    srcs = glob(["*.bzl"]),
)

genrule(
    name = "empty",
    outs = ["empty.bzl"],
    cmd = "touch $@",
)

REPO_BZL_FILES = [
    "git",
    "http",
    "utils",
]

[stardoc(
    name = "%s_raw_md" % (name,),
    out = "%s_raw.md" % (name,),
    input = "%s.bzl" % (name,),
    deps = [":lib"],
) for name in REPO_BZL_FILES]

[genrule(
    name = "%s_md" % (name,),
    srcs = [
        ":preamb.md",
        ":%s_raw_md" % (name,),
    ],
    outs = ["%s.md" % (name,)],
    cmd = "cat $(SRCS) > $@",
) for name in REPO_BZL_FILES]

pkg_tar(
    name = "doc",
    srcs = ["%s_md" % (name,) for name in REPO_BZL_FILES] + ["index.md"],
    strip_prefix = "/tools/build_defs/repo",
    visibility = ["//site:__subpackages__"],
)
