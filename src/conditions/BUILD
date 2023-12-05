load("@bazel_skylib//lib:selects.bzl", "selects")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = [
        "//src:__pkg__",
        "//src/test/shell/bazel:__pkg__",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = ["BUILD.tools"],
    visibility = ["//src:__pkg__"],
)

config_setting(
    name = "linux",
    constraint_values = ["@platforms//os:linux"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_aarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_arm",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:arm",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:ppc",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_ppc64le",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:ppc",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_s390x",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:s390x",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_mips64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:mips64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_riscv64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:riscv64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_loongarch64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:loongarch64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "linux_x86_64",
    constraint_values = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin",
    constraint_values = ["@platforms//os:macos"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_x86_64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64_constraint",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
    ],
)

config_setting(
    name = "darwin_arm64_flag",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    values = {"cpu": "darwin_arm64"},
)

# Workaround for an issue where the dummy cc toolchain isn't being picked up
# when cross-compile from darwin_x86_64 to darwin_arm64 cpu.
# TODO(https://github.com/bazelbuild/bazel/issues/12655): Remove the flag based
# select when the issue is resolved.
selects.config_setting_group(
    name = "darwin_arm64",
    match_any = [
        ":darwin_arm64_constraint",
        ":darwin_arm64_flag",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "darwin_arm64e",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64e",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "iphonesdk",
    values = {"define": "IPHONE_SDK=1"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "freebsd",
    constraint_values = ["@platforms//os:freebsd"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "openbsd",
    constraint_values = ["@platforms//os:openbsd"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows",
    constraint_values = ["@platforms//os:windows"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_x86_64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_arm64_constraint",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:arm64",
    ],
)

# TODO(https://github.com/bazelbuild/bazel/issues/7260)
# Remove the flag after toolchain cc resolution is fixed
config_setting(
    name = "windows_x64_arm64_flag",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    values = {"cpu": "x64_arm64_windows"},
)

config_setting(
    name = "windows_arm64_flag",
    values = {"cpu": "arm64_windows"},
)

selects.config_setting_group(
    name = "windows_arm64",
    match_any = [
        ":windows_arm64_constraint",
        ":windows_arm64_flag",
        ":windows_x64_arm64_flag",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    constraint_values = ["@platforms//cpu:arm"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "remote",
    values = {"define": "EXECUTOR=remote"},
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debian_build",
    values = {
        "define": "distribution=debian",
    },
    visibility = ["//visibility:public"],
)
