package(default_visibility = ["//visibility:public"])

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD.java_tools_prebuilt_darwin_x86_64",
        "BUILD.tools",
        "default_java_toolchain.bzl",
        "java_toolchain_alias.bzl",
        "launcher_flag_alias.bzl",
        "local_java_repository.bzl",
        "proguard_allowlister.cc",
        "proguard_allowlister_lib.cc",
        "proguard_allowlister_lib.h",
        "proguard_allowlister_test.cc",
        "proguard_allowlister_test_input.pgcfg",
        "proguard_allowlister_test_lib.cc",
        "proguard_allowlister_test_lib.h",
        "remote_java_repository.bzl",
        "toolchain_utils.bzl",
    ],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

filegroup(
    name = "embedded_tools",
    srcs = [
        "BUILD.tools",
        "default_java_toolchain.bzl",
        "java_toolchain_alias.bzl",
        "launcher_flag_alias.bzl",
        "local_java_repository.bzl",
        "proguard_allowlister.cc",
        "proguard_allowlister_lib.cc",
        "proguard_allowlister_lib.h",
        "proguard_allowlister_test.cc",
        "proguard_allowlister_test_input.pgcfg",
        "proguard_allowlister_test_lib.cc",
        "proguard_allowlister_test_lib.h",
        "remote_java_repository.bzl",
        "toolchain_utils.bzl",
    ],
    visibility = ["//tools:__pkg__"],
)

alias(
    name = "java_tools_prebuilt_build_file",
    actual = select({
        "//src/conditions:darwin_x86_64": ":BUILD.java_tools_prebuilt_darwin_x86_64",
        "//conditions:default": ":BUILD.java_tools_prebuilt",
    }),
    visibility = ["//src:__pkg__"],
)

exports_files(
    [
        "BUILD.java_tools",
    ],
    visibility = ["//src:__pkg__"],
)
