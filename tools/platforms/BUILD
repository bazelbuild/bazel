# Deprecated. Do not use. Use targets in @platforms//cpu and @platforms//os.
# This package will be removed by
# --incompatible_use_platforms_repo_for_constraints. See
# https://github.com/bazelbuild/bazel/issues/8622 for details.
package(
    default_visibility = ["//visibility:public"],
)

filegroup(
    name = "package-srcs",
    srcs = [
        "BUILD.tools",
    ],
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
)

alias(
    name = "cpu",
    actual = "@platforms//cpu:cpu",
)

alias(
    name = "x86_32",
    actual = "@platforms//cpu:x86_32",
)

alias(
    name = "x86_64",
    actual = "@platforms//cpu:x86_64",
)

alias(
    name = "ppc",
    actual = "@platforms//cpu:ppc",
)

alias(
    name = "arm",
    actual = "@platforms//cpu:arm",
)

alias(
    name = "aarch64",
    actual = "@platforms//cpu:aarch64",
)

alias(
    name = "s390x",
    actual = "@platforms//cpu:s390x",
)

alias(
    name = "mips64",
    actual = "@platforms//cpu:mips64",
)

alias(
    name = "riscv64",
    actual = "@platforms//cpu:riscv64",
)

alias(
    name = "os",
    actual = "@platforms//os:os",
)

alias(
    name = "osx",
    actual = "@platforms//os:osx",
)

alias(
    name = "ios",
    actual = "@platforms//os:ios",
)

alias(
    name = "freebsd",
    actual = "@platforms//os:freebsd",
)

alias(
    name = "openbsd",
    actual = "@platforms//os:openbsd",
)

alias(
    name = "android",
    actual = "@platforms//os:android",
)

alias(
    name = "linux",
    actual = "@platforms//os:linux",
)

alias(
    name = "windows",
    actual = "@platforms//os:windows",
)

# A default platform with nothing defined.
platform(name = "default_platform")
