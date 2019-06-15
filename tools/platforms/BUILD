load(":fail_with_incompatible_use_platforms_repo_for_constraints.bzl", "fail_with_incompatible_use_platforms_repo_for_constraints")

package(
    default_visibility = ["//visibility:public"],
)

fail_with_incompatible_use_platforms_repo_for_constraints(
    name = "fail_with_incompatible_use_platforms_repo_for_constraints",
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

config_setting(
    name = "using_platforms_not_bazel_tools",
    values = {"incompatible_use_platforms_repo_for_constraints": "true"},
)

alias(
    name = "cpu",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:cpu",
    }),
)

alias(
    name = "x86_32",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:x86_32",
    }),
)

alias(
    name = "x86_64",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:x86_64",
    }),
)

alias(
    name = "ppc",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:ppc",
    }),
)

alias(
    name = "arm",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:arm",
    }),
)

alias(
    name = "aarch64",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:aarch64",
    }),
)

alias(
    name = "s390x",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//cpu:s390x",
    }),
)

alias(
    name = "os",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:os",
    }),
)

alias(
    name = "osx",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:osx",
    }),
)

alias(
    name = "ios",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:ios",
    }),
)

alias(
    name = "freebsd",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:freebsd",
    }),
)

alias(
    name = "android",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:android",
    }),
)

alias(
    name = "linux",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:linux",
    }),
)

alias(
    name = "windows",
    actual = select({
        ":using_platforms_not_bazel_tools": ":fail_with_incompatible_use_platforms_repo_for_constraints",
        "//conditions:default": "@platforms//os:windows",
    }),
)

# A default platform with nothing defined.
platform(name = "default_platform")

# A default platform referring to the host system. This only exists for
# internal build configurations, and so shouldn't be accessed by other packages.
platform(
    name = "host_platform",
    constraint_values = [
    ],
    cpu_constraints = [
        "@platforms//cpu:x86_32",
        "@platforms//cpu:x86_64",
        "@platforms//cpu:ppc",
        "@platforms//cpu:arm",
        "@platforms//cpu:aarch64",
        "@platforms//cpu:s390x",
    ],
    host_platform = True,
    os_constraints = [
        "@platforms//os:osx",
        "@platforms//os:freebsd",
        "@platforms//os:linux",
        "@platforms//os:windows",
    ],
)

platform(
    name = "target_platform",
    constraint_values = [
    ],
    cpu_constraints = [
        "@platforms//cpu:x86_32",
        "@platforms//cpu:x86_64",
        "@platforms//cpu:ppc",
        "@platforms//cpu:arm",
        "@platforms//cpu:aarch64",
        "@platforms//cpu:s390x",
    ],
    os_constraints = [
        "@platforms//os:osx",
        "@platforms//os:freebsd",
        "@platforms//os:linux",
        "@platforms//os:windows",
    ],
    target_platform = True,
)
