licenses(["notice"])  # Apache 2.0

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
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/gflags",
    ],
)

py_binary(
    name = "make_deb",
    srcs = ["make_deb.py"],
    srcs_version = "PY2AND3",
    visibility = ["//visibility:public"],
    deps = [
        ":archive",
        "//third_party/py/gflags",
    ],
)

# Used by pkg_rpm in rpm.bzl.
py_binary(
    name = "make_rpm",
    srcs = ["make_rpm.py"],
    visibility = ["//visibility:public"],
    deps = [
        "//third_party/py/gflags",
    ],
)

py_test(
    name = "make_rpm_test",
    srcs = ["make_rpm_test.py"],
    # rpmbuild is not available in windows
    tags = [
        "no_windows",
    ],
    deps = [
        ":make_rpm",
    ],
)

# tests
load("//tools/build_defs/pkg:pkg.bzl", "pkg_deb", "pkg_tar")

genrule(
    name = "generate_files",
    outs = [
        "etc/nsswitch.conf",
        "usr/titi",
    ],
    cmd = "for i in $(OUTS); do echo 1 >$$i; done",
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

pkg_deb(
    name = "test-deb",
    conffiles = [
        "/etc/nsswitch.conf",
        "/etc/other",
    ],
    data = ":test-tar-gz.tar.gz",
    depends = [
        "dep1",
        "dep2",
    ],
    description = "toto",
    distribution = "trusty",
    maintainer = "someone@somewhere.com",
    make_deb = ":make_deb",
    package = "titi",
    urgency = "low",
    version = "test",
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
