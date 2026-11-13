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
        "@platforms//cpu:ppc64le",
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
    name = "darwin_arm64",
    constraint_values = [
        "@platforms//os:macos",
        "@platforms//cpu:arm64",
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
    name = "windows_arm64",
    constraint_values = [
        "@platforms//os:windows",
        "@platforms//cpu:arm64",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_clang_cl",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "clang-cl"},
)

config_setting(
    name = "windows_msvc_cl",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "msvc-cl"},
)

selects.config_setting_group(
    name = "windows_msvc_like",
    match_any = [
        ":windows_clang_cl",
        ":windows_msvc_cl",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "windows_clang",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "clang"},
)

config_setting(
    name = "windows_gcc",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "gcc"},
)

config_setting(
    name = "windows_mingw_gcc",
    constraint_values = ["@platforms//os:windows"],
    flag_values = {"@rules_cc//cc/compiler:compiler": "mingw-gcc"},
)

# Compilers targeting Windows that use a GCC-style driver (MinGW), which expects
# -l<lib> rather than -DEFAULTLIB:<lib>.lib to link against system libraries.
selects.config_setting_group(
    name = "windows_gcc_like",
    match_any = [
        ":windows_clang",
        ":windows_gcc",
        ":windows_mingw_gcc",
    ],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "arm",
    constraint_values = ["@platforms//cpu:arm"],
    visibility = ["//visibility:public"],
)

config_setting(
    name = "debian_build",
    values = {
        "define": "distribution=debian",
    },
    visibility = ["//visibility:public"],
)
