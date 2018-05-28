package(default_visibility = ["//visibility:private"])

filegroup(
    name = "srcs",
    srcs = glob(
        ["**"],
        exclude = [
            "*~",
            ".*",
        ],
    ),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "sh_configure.bzl",
        "sh_toolchain.bzl",
    ],
    visibility = ["//tools:__pkg__"],
)

# Do not depend on this rule.
# Use @bazel_tools//tools/sh:toolchain_type instead.
#
# This rule is available exclusively for the Bazel bootstrapping process.
#
# As part of bootstrapping Bazel, we set up a mock @bazel_tools repo, by
# creating a directory with a WORKSPACE file and symlinking the real tools
# directory into it.  This approach however won't use the BUILD.tools files as
# BUILD files, so the fake repo won't be exactly the same as the real one.  To
# work around this issue without having to symlink each file in the entire
# tools/** tree and symlinking BUILD.tools files as BUILD files, the
# bootstrapper just symlinks the top tools directory. Therefore to bootstrap
# Bazel, we need those targets in BUILD.tools files that Bazel need to also
# exist in BUILD files.
toolchain_type(
    name = "toolchain_type",
    visibility = ["//visibility:public"],
)
