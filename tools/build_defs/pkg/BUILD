licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
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
    build_tar = ":build_tar",
    extension = "tar%s" % ext,
    files = [
        ":etc/nsswitch.conf",
        ":usr/titi",
    ],
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
    build_tar = ":build_tar",
    files = [
        ":etc/nsswitch.conf",
    ],
    strip_prefix = "",
)

pkg_tar(
    name = "test-tar-strip_prefix-none",
    build_tar = ":build_tar",
    files = [
        ":etc/nsswitch.conf",
    ],
)

pkg_tar(
    name = "test-tar-strip_prefix-etc",
    build_tar = ":build_tar",
    files = [
        ":etc/nsswitch.conf",
    ],
    strip_prefix = "etc",
)

pkg_tar(
    name = "test-tar-strip_prefix-dot",
    build_tar = ":build_tar",
    files = [
        ":etc/nsswitch.conf",
    ],
    strip_prefix = ".",
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
    deps = [
        "//src/test/shell:bashunit",
    ],
)
