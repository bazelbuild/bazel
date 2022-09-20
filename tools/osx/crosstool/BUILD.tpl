package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/osx/crosstool:osx_archs.bzl", "OSX_TOOLS_ARCHS")
load("@rules_cc//cc:defs.bzl", "cc_toolchain_suite", "cc_library")
load(":armeabi_cc_toolchain_config.bzl", "armeabi_cc_toolchain_config")
load(":cc_toolchain_config.bzl", "cc_toolchain_config")

# Reexporting osx_arch.bzl for backwards compatibility
# Originally this file was present in @local_config_cc, but with the split in
# https://github.com/bazelbuild/bazel/pull/8459 we had to move the file to
# @local_config_cc_toolchains. This alias is there to keep the code backwards
# compatible (and serves no other purpose).
alias(name = "osx_archs.bzl", actual = "@bazel_tools//tools/osx/crosstool:osx_archs.bzl")

CC_TOOLCHAINS = [(
    cpu + "|clang",
    ":cc-compiler-" + cpu,
) for cpu in OSX_TOOLS_ARCHS] + [(
    cpu,
    ":cc-compiler-" + cpu,
) for cpu in OSX_TOOLS_ARCHS] + [
    ("k8|clang", ":cc-compiler-darwin_x86_64"),
    ("darwin|clang", ":cc-compiler-darwin_x86_64"),
    ("k8", ":cc-compiler-darwin_x86_64"),
    ("darwin", ":cc-compiler-darwin_x86_64"),
    ("armeabi-v7a|compiler", ":cc-compiler-armeabi-v7a"),
    ("armeabi-v7a", ":cc-compiler-armeabi-v7a"),
]

cc_library(
    name = "malloc",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "cc_wrapper",
    srcs = ["cc_wrapper.sh"],
)

cc_toolchain_suite(
    name = "toolchain",
    toolchains = dict(CC_TOOLCHAINS),
)

[
    filegroup(
        name = "osx_tools_" + arch,
        srcs = [
            ":cc_wrapper",
            ":libtool",
            ":libtool_check_unique",
            ":make_hashed_objlist.py",
            ":wrapped_clang",
            ":wrapped_clang_pp",
            ":xcrunwrapper.sh",
        ],
    )
    for arch in OSX_TOOLS_ARCHS
]

[
    apple_cc_toolchain(
        name = "cc-compiler-" + arch,
        all_files = ":osx_tools_" + arch,
        ar_files = ":osx_tools_" + arch,
        as_files = ":osx_tools_" + arch,
        compiler_files = ":osx_tools_" + arch,
        dwp_files = ":empty",
        linker_files = ":osx_tools_" + arch,
        objcopy_files = ":empty",
        strip_files = ":osx_tools_" + arch,
        supports_param_files = 1,
        toolchain_config = arch,
        toolchain_identifier = arch,
    )
    for arch in OSX_TOOLS_ARCHS
]

# When xcode_locator fails and causes cc_autoconf_toolchains to fall back
# to the non-Xcode C++ toolchain, it uses the legacy cpu value to refer to
# the toolchain, which is "darwin" for x86_64 macOS.
alias(
    name = "cc-compiler-darwin",
    actual = ":cc-compiler-darwin_x86_64",
)

[
    cc_toolchain_config(
        name = arch,
        compiler = "clang",
        cpu = arch,
        cxx_builtin_include_directories = [
%{cxx_builtin_include_directories}
        ],
        tool_paths_overrides = {%{tool_paths_overrides}},
    )
    for arch in OSX_TOOLS_ARCHS
]

# Android tooling requires a default toolchain for the armeabi-v7a cpu.
cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    toolchain_identifier = "stub_armeabi-v7a",
    toolchain_config = ":stub_armeabi-v7a",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
)

armeabi_cc_toolchain_config(name = "stub_armeabi-v7a")
