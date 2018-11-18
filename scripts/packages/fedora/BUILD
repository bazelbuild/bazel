package(default_visibility = ["//scripts/packages:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

load("//tools/build_defs/pkg:rpm.bzl", "pkg_rpm")

pkg_rpm(
    name = "bazel",
    architecture = "x86_64",
    changelog = "//:changelog-file",
    data = [
        "//scripts:bash_completion",
        "//scripts/zsh_completion:_bazel",
        "//scripts/packages:bazel.bazelrc",
        "//scripts/packages:without-jdk/bazel",
        "//scripts/packages:without-jdk/bazel-real",
    ],
    spec_file = "bazel.spec",
    version_file = "//scripts/packages:version.txt",
)
