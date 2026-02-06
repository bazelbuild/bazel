package(default_visibility = ["//visibility:public"])

load("@rules_java//toolchains:default_java_toolchain.bzl", "default_java_toolchain")

filegroup(
    name = "BUILD-jdk",
    srcs = [":BUILD.tools"],
)

# JDK 27 EA toolchain for testing - compiles Java 21 code using JDK 27
default_java_toolchain(
    name = "toolchain_jdk_27",
    configuration = {
        "java_runtime": "@remotejdk27_linux//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:x86_64",
    ],
    source_version = "21",
    target_version = "21",
)

default_java_toolchain(
    name = "toolchain_jdk_27_linux_aarch64",
    configuration = {
        "java_runtime": "@remotejdk27_linux_aarch64//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:linux",
        "@platforms//cpu:aarch64",
    ],
    source_version = "21",
    target_version = "21",
)

default_java_toolchain(
    name = "toolchain_jdk_27_macos",
    configuration = {
        "java_runtime": "@remotejdk27_macos//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:x86_64",
    ],
    source_version = "21",
    target_version = "21",
)

default_java_toolchain(
    name = "toolchain_jdk_27_macos_aarch64",
    configuration = {
        "java_runtime": "@remotejdk27_macos_aarch64//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:macos",
        "@platforms//cpu:aarch64",
    ],
    source_version = "21",
    target_version = "21",
)

default_java_toolchain(
    name = "toolchain_jdk_27_win",
    configuration = {
        "java_runtime": "@remotejdk27_win//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:windows",
        "@platforms//cpu:x86_64",
    ],
    source_version = "21",
    target_version = "21",
)

default_java_toolchain(
    name = "toolchain_jdk_27_win_arm64",
    configuration = {
        "java_runtime": "@remotejdk27_win_arm64//:jdk",
    },
    exec_compatible_with = [
        "@platforms//os:windows",
        "@platforms//cpu:arm64",
    ],
    source_version = "21",
    target_version = "21",
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD-jdk",  # Tools are build from the workspace for tests.
        "default_java_toolchain.bzl",
        "java_toolchain_alias.bzl",
        "launcher_flag_alias.bzl",
        "local_java_repository.bzl",
        "nosystemjdk/README",
        "proguard_allowlister.py",
        "proguard_allowlister_test.py",
        "proguard_allowlister_test_input.pgcfg",
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
    name = "package-srcs",
    srcs = glob(["**"]),
)
