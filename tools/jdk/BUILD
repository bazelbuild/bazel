load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")

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
        "remote_java_repository.bzl",
        "toolchain_utils.bzl",
    ],
    visibility = ["//tools:__pkg__"],
)

cc_library(
    name = "proguard_allowlister_test_lib",
    testonly = 1,
    srcs = [
        "proguard_allowlister_lib.cc",
        "proguard_allowlister_lib.h",
        "proguard_allowlister_test_lib.cc",
    ],
    hdrs = ["proguard_allowlister_test_lib.h"],
    deps = ["@rules_cc//cc/runfiles"],
)

cc_test(
    name = "proguard_allowlister_test",
    srcs = [
        "proguard_allowlister_test.cc",
    ],
    data = ["proguard_allowlister_test_input.pgcfg"],
    env = {
        "TEST_INPUT": "$(rlocationpath :proguard_allowlister_test_input.pgcfg)",
    },
    deps = [
        ":proguard_allowlister_test_lib",
    ],
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
