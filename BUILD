package(default_visibility = ["//scripts/release:__pkg__"])

filegroup(
    name = "git",
    srcs = glob([".git/**"]),
)

filegroup(
    name = "dummy",
    visibility = ["//visibility:public"],
)

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "bazel-*/**",
            "output/**",
            ".*/**",
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

load("//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "bazel-srcs",
    files = [":srcs"],
    strip_prefix = ".",
    # Public but bazel-only visibility.
    visibility = ["//:__subpackages__"],
)

load("//tools/build_rules/go:def.bzl", "go_prefix")

go_prefix("github.com/bazelbuild/bazel")
