# Bazel - Google's Build System

package(default_visibility = ["//scripts/release:__pkg__"])

exports_files(["LICENSE"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["*"],
        exclude = [
            "bazel-*",  # convenience symlinks
            "out",  # IntelliJ with setup-intellij.sh
            "output",  # output of compile.sh
            ".*",  # mainly .git* files
        ],
    ) + [
        "//examples:srcs",
        "//scripts:srcs",
        "//site:srcs",
        "//src:srcs",
        "//tools:srcs",
        "//third_party:srcs",
    ],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "git",
    srcs = glob([".git/**"]),
)

filegroup(
    name = "dummy",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "workspace-file",
    srcs = [":WORKSPACE"],
    visibility = [
        "//src/test/docker:__pkg__",
        "//src/test/shell/bazel:__subpackages__",
    ],
)

filegroup(
    name = "changelog-file",
    srcs = [":CHANGELOG.md"],
    visibility = [
        "//scripts/packages:__subpackages__",
    ],
)

filegroup(
    name = "bootstrap-derived-java-srcs",
    srcs = glob(["derived/**/*.java"]),
    visibility = ["//:__subpackages__"],
)

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "bazel-srcs",
    srcs = [":srcs"],
    strip_prefix = ".",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

py_binary(
    name = "combine_distfiles",
    srcs = ["combine_distfiles.py"],
    visibility = ["//visibility:private"],
    deps = ["//src:create_embedded_tools_lib"],
)

genrule(
    name = "bazel-distfile",
    srcs = [
        ":bazel-srcs",
        "//src:derived_java_srcs",
    ],
    outs = ["bazel-distfile.zip"],
    cmd = "$(location :combine_distfiles) $@ $(SRCS)",
    tools = [":combine_distfiles"],
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

genrule(
    name = "bazel-distfile-tar",
    srcs = [
        ":bazel-srcs",
        "//src:derived_java_srcs",
    ],
    outs = ["bazel-distfile.tar"],
    cmd = "$(location :combine_distfiles_to_tar.sh) $@ $(SRCS)",
    tools = ["combine_distfiles_to_tar.sh"],
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

# This is a workaround for fetching Bazel toolchains, for remote execution.
# See https://github.com/bazelbuild/bazel/issues/3246.
# Will be removed once toolchain fetching is supported.
filegroup(
    name = "dummy_toolchain_reference",
    srcs = ["@bazel_toolchains//configs/debian8_clang/0.1.0:empty"],
    visibility = ["//visibility:public"],
)
