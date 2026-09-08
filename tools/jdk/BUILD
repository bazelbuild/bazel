package(default_visibility = ["//visibility:public"])

filegroup(
    name = "BUILD-jdk",
    srcs = [":BUILD.tools"],
)

filegroup(
    name = "srcs",
    srcs = [
        "BUILD",
        "BUILD.java_tools",
        "BUILD.java_tools_prebuilt",
        "BUILD.java_tools_prebuilt_darwin_x86_64",
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
