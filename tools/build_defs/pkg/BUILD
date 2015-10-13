licenses(["notice"])  # Apache 2.0

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//tools:__pkg__"],
)

# Used by pkg_deb
py_library(
    name = "archive",
    srcs = ["archive.py"],
    visibility = [
        "//tools/build_defs/docker:__subpackages__",
    ],
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
    deps = [":archive"],
)

py_binary(
    name = "build_tar",
    srcs = ["build_tar.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@bazel_tools//third_party/py/gflags",
        ":archive",
    ],
)

py_binary(
    name = "make_deb",
    srcs = ["make_deb.py"],
    visibility = ["//visibility:public"],
    deps = [
        "@bazel_tools//third_party/py/gflags",
        ":archive",
    ],
)

# tests
load("/tools/build_defs/pkg/pkg", "pkg_deb", "pkg_tar")

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
    extension = "tar%s" % ext,
    files = [
        ":etc/nsswitch.conf",
        ":usr/titi",
    ],
    mode = "0644",
    modes = {"usr/titi": "0755"},
    package_dir = "/",
    strip_prefix = ".",
    symlinks = {"usr/bin/java": "/path/to/bin/java"},
) for ext in [
    "",
    ".gz",
    ".bz2",
]]

pkg_deb(
    name = "test-deb",
    data = ":test-tar-gz.tar.gz",
    depends = [
        "dep1",
        "dep2",
    ],
    description = "toto",
    maintainer = "someone@somewhere.com",
    package = "titi",
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
    ],
    deps = [
        "//src/test/shell:bashunit",
    ],
)
