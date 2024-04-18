load("@rules_pkg//:rpm.bzl", "pkg_rpm")

package(default_visibility = ["//scripts/packages:__subpackages__"])

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

pkg_rpm(
    name = "bazel",
    architecture = "x86_64",
    changelog = "//:changelog-file",
    data = [
        "//scripts:bash_completion",
        "//scripts/packages:bazel.bazelrc",
        "//scripts/packages:without-jdk/bazel",
        "//scripts/packages:without-jdk/bazel-real",
        "//scripts/zsh_completion:_bazel",
    ],
    spec_file = "bazel.spec",
    version_file = "//scripts/packages:version.txt",
)
