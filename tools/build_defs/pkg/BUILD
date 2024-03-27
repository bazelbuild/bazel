load("@rules_python//python:defs.bzl", "py_binary", "py_library", "py_test")

# -*- coding: utf-8 -*-
load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

licenses(["notice"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD",
        "archive.py",
        "build_tar.py",
        "path.bzl",
        "pkg.bzl",
    ],
    visibility = ["//tools/build_defs:__pkg__"],
)

py_library(
    name = "archive",
    srcs = ["archive.py"],
    srcs_version = "PY3",
)

filegroup(
    name = "archive_testdata",
    srcs = glob(
        ["testdata/**"],
        allow_empty = True,
    ),
)

py_test(
    name = "archive_test",
    srcs = [
        "archive_test.py",
    ],
    data = [":archive_testdata"],
    python_version = "PY3",
    srcs_version = "PY3",
    tags = [
        # TODO(laszlocsomor): fix on Windows or describe why it cannot pass.
        "no_windows",
    ],
    deps = [":archive"],
)

py_test(
    name = "path_test",
    srcs = ["path_test.py"],
    data = ["path.bzl"],
    python_version = "PY3",
    srcs_version = "PY3",
)

py_binary(
    name = "build_tar",
    srcs = ["build_tar.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
    ],
)

# tests

genrule(
    name = "generate_files",
    outs = [
        "etc/nsswitch.conf",
        "usr/titi",
    ],
    cmd = "for i in $(OUTS); do echo 1 >$$i; done",
)

pkg_tar(
    name = "test-tar",
    srcs = [
        ":etc/nsswitch.conf",
        ":usr/titi",
    ],
    build_tar = ":build_tar",
    package_dir = "/",
)

sh_test(
    name = "build_test",
    size = "medium",
    srcs = [
        "build_test.sh",
    ],
    data = [
        "testenv.sh",
        ":test-tar.tar",
    ],
    tags = [
        # TODO(laszlocsomor): fix on Windows or describe why it cannot pass.
        "no_windows",
    ],
    deps = [
        "//src/test/shell:bashunit",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools/build_defs:__pkg__"],
)
