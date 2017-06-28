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
        "//src/test/shell/bazel:__subpackages__",
        "//src/test/docker:__pkg__",
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
    files = [":srcs"],
    strip_prefix = ".",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

genrule(
    name = "bazel-distfile",
    srcs = [
        ":bazel-srcs",
        "//src:derived_java_srcs",
    ],
    outs = ["bazel-distfile.zip"],
    cmd = "$(location :combine_distfiles.sh) $@ $(SRCS)",
    tools = ["combine_distfiles.sh"],
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
    cmd = "env USE_TAR=YES $(location :combine_distfiles.sh) $@ $(SRCS)",
    tools = ["combine_distfiles.sh"],
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)
