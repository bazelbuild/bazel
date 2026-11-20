load("@rules_cc//cc:cc_binary.bzl", "cc_binary")
load("@rules_cc//cc:cc_library.bzl", "cc_library")
load("@rules_cc//cc:cc_test.bzl", "cc_test")
load("//src:release_archive.bzl", "release_archive")

package(default_visibility = ["//visibility:public"])

PROGUARD_ALLOWLISTER_SOURCES = [
    # keep sorted
    "proguard_allowlister.cc",
    "proguard_allowlister_lib.cc",
    "proguard_allowlister_lib.h",
]

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD.java_tools_prebuilt_darwin_x86_64",
        "BUILD.tools",
        "default_java_toolchain.bzl",
        "launcher_flag_alias.bzl",
        "local_java_repository.bzl",
        "proguard_allowlister.cc",
        "proguard_allowlister_lib.cc",
        "proguard_allowlister_lib.h",
        "proguard_allowlister_test.cc",
        "proguard_allowlister_test_input.pgcfg",
        "proguard_allowlister_test_lib.cc",
        "proguard_allowlister_test_lib.h",
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
        "launcher_flag_alias.bzl",
        "local_java_repository.bzl",
        "proguard_allowlister.cc",
        "proguard_allowlister_lib.cc",
        "proguard_allowlister_lib.h",
        "toolchain_utils.bzl",
    ],
    visibility = ["//tools:__pkg__"],
)

# The proguard allowlister is also shipped with java_tools, both as sources (to
# build it from source on platforms without a prebuilt binary) and as a prebuilt
# binary, so that the Java toolchain doesn't require a C++ toolchain for the
# execution platform. Keep in sync with :proguard_allowlister in BUILD.tools.
cc_binary(
    name = "proguard_allowlister",
    srcs = PROGUARD_ALLOWLISTER_SOURCES,
    visibility = ["//visibility:private"],
)

filegroup(
    name = "proguard_allowlister_srcs",
    srcs = PROGUARD_ALLOWLISTER_SOURCES,
    visibility = ["//src:__pkg__"],
)

release_archive(
    name = "proguard_allowlister_zip",
    srcs = PROGUARD_ALLOWLISTER_SOURCES,
    package_dir = "java_tools/tools/jdk",
    visibility = ["//src:__pkg__"],
)

release_archive(
    name = "proguard_allowlister_deploy_zip",
    srcs = [":proguard_allowlister"],
    package_dir = "java_tools/tools/jdk",
    visibility = ["//src:__pkg__"],
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
