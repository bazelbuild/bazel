# -*- coding: utf-8 -*-
load("//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")
load("//tools/config:common_settings.bzl", "bool_flag")
load("//tools/python:private/defs.bzl", "py_binary", "py_library", "py_test")
load("//tools/python:private/version.bzl", "PY_BINARY_VERSION")

licenses(["notice"])  # Apache 2.0

bool_flag(
    name = "incompatible_no_build_defs_pkg",
    build_setting_default = False,
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools/build_defs:__pkg__"],
)

exports_files(
    ["README.md"],
    visibility = ["//site:__pkg__"],
)

# Used by pkg_deb
py_library(
    name = "archive",
    srcs = ["archive.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "archive_testdata",
    srcs = glob(["testdata/**"]),
)

py_test(
    name = "archive_test",
    srcs = [
        "archive_test.py",
        "testenv.py",
    ],
    data = [":archive_testdata"],
    python_version = PY_BINARY_VERSION,
    srcs_version = "PY2AND3",
    tags = [
        # archive.py requires xzcat, which is not available by default on Mac
        "noci",
        # TODO(laszlocsomor): fix on Windows or describe why it cannot pass.
        "no_windows",
    ],
    deps = [":archive"],
)

py_test(
    name = "path_test",
    srcs = ["path_test.py"],
    data = ["path.bzl"],
    srcs_version = "PY2AND3",
)

py_binary(
    name = "build_tar",
    srcs = ["build_tar.py"],
    python_version = PY_BINARY_VERSION,
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/abseil",
    ],
)

py_binary(
    name = "make_deb",
    srcs = ["make_deb.py"],
    deprecation = "The internal version of make_deb is deprecated. Please " +
                  "use the replacement for pkg_deb from " +
                  "https://github.com/bazelbuild/rules_pkg/blob/master/pkg.",
    python_version = PY_BINARY_VERSION,
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/abseil",
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

pkg_deb(
    name = "test-deb",
    built_using = "some_test_data (0.1.2)",
    conffiles = [
        "/etc/nsswitch.conf",
        "/etc/other",
    ],
    config = ":testdata/config",
    data = ":test-tar-gz.tar.gz",
    depends = [
        "dep1",
        "dep2",
    ],
    description = "toto ®, Й, ק ,م, ๗, あ, 叶, 葉, 말, ü and é",
    distribution = "trusty",
    maintainer = "soméone@somewhere.com",
    make_deb = ":make_deb",
    package = "titi",
    templates = ":testdata/templates",
    urgency = "low",
    version = "test",
)

[pkg_tar(
    name = "test-tar-%s" % ext[1:],
    srcs = [
        ":etc/nsswitch.conf",
        ":usr/titi",
    ],
    build_tar = ":build_tar",
    extension = "tar%s" % ext,
    mode = "0644",
    modes = {"usr/titi": "0755"},
    owner = "42.24",
    ownername = "titi.tata",
    ownernames = {"etc/nsswitch.conf": "tata.titi"},
    owners = {"etc/nsswitch.conf": "24.42"},
    package_dir = "/",
    strip_prefix = ".",
    symlinks = {"usr/bin/java": "/path/to/bin/java"},
) for ext in [
    "",
    ".gz",
    ".bz2",
    ".xz",  # This will breaks if xzcat is not installed
]]

[pkg_tar(
    name = "test-tar-inclusion-%s" % ext,
    build_tar = ":build_tar",
    deps = [":test-tar-%s" % ext],
) for ext in [
    "",
    "gz",
    "bz2",
    "xz",
]]

pkg_tar(
    name = "test-tar-strip_prefix-empty",
    srcs = [
        ":etc/nsswitch.conf",
    ],
    build_tar = ":build_tar",
    strip_prefix = "",
)

pkg_tar(
    name = "test-tar-strip_prefix-none",
    srcs = [
        ":etc/nsswitch.conf",
    ],
    build_tar = ":build_tar",
)

pkg_tar(
    name = "test-tar-strip_prefix-etc",
    srcs = [
        ":etc/nsswitch.conf",
    ],
    build_tar = ":build_tar",
    strip_prefix = "etc",
)

pkg_tar(
    name = "test-tar-strip_prefix-dot",
    srcs = [
        ":etc/nsswitch.conf",
    ],
    build_tar = ":build_tar",
    strip_prefix = ".",
)

pkg_tar(
    name = "test-tar-files_dict",
    build_tar = ":build_tar",
    files = {
        ":etc/nsswitch.conf": "not-etc/mapped-filename.conf",
    },
)

pkg_tar(
    name = "test-tar-empty_files",
    build_tar = ":build_tar",
    empty_files = [
        "/a",
        "/b",
    ],
    mode = "0o777",
)

pkg_tar(
    name = "test-tar-empty_dirs",
    build_tar = ":build_tar",
    empty_dirs = [
        "/tmp",
        "/pmt",
    ],
    mode = "0o777",
)

pkg_tar(
    name = "test-tar-mtime",
    srcs = [
        ":etc/nsswitch.conf",
    ],
    build_tar = ":build_tar",
    mtime = 946684740,  # 1999-12-31, 23:59
    portable_mtime = False,
)

sh_test(
    name = "build_test",
    size = "medium",
    srcs = [
        "build_test.sh",
    ],
    data = [
        "testenv.sh",
        ":test-deb.deb",
        ":test-tar-.tar",
        ":test-tar-bz2.tar.bz2",
        ":test-tar-empty_dirs.tar",
        ":test-tar-empty_files.tar",
        ":test-tar-files_dict.tar",
        ":test-tar-gz.tar.gz",
        ":test-tar-inclusion-.tar",
        ":test-tar-inclusion-bz2.tar",
        ":test-tar-inclusion-gz.tar",
        ":test-tar-inclusion-xz.tar",
        ":test-tar-mtime.tar",
        ":test-tar-strip_prefix-dot.tar",
        ":test-tar-strip_prefix-empty.tar",
        ":test-tar-strip_prefix-etc.tar",
        ":test-tar-strip_prefix-none.tar",
        ":test-tar-xz.tar.xz",
        ":titi_test_all.changes",
    ],
    tags = [
        # archive.py requires xzcat, which is not available by default on Mac
        "noci",
        # TODO(laszlocsomor): fix on Windows or describe why it cannot pass.
        "no_windows",
    ],
    deps = [
        "//src/test/shell:bashunit",
    ],
)

test_suite(
    name = "windows_tests",
    tags = [
        "-no_windows",
        "-slow",
    ],
    visibility = ["//visibility:private"],
)

test_suite(
    name = "all_windows_tests",
    tests = [":windows_tests"],
    visibility = ["//tools/build_defs:__pkg__"],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools/build_defs:__pkg__"],
)
