# Copyright 2018 The Bazel Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

load(":cc_toolchain_config.bzl", "cc_toolchain_config")
load("@rules_cc//cc:defs.bzl", "cc_library", "cc_toolchain", "cc_toolchain_suite")

# The toolchain type used to distinguish cc toolchains.
toolchain_type(name = "toolchain_type")

# It is frequently necessary to constrain platforms based on the cc compiler type.
constraint_setting(name = "cc_compiler")

constraint_value(
    name = "clang",
    constraint_setting = ":cc_compiler",
)

constraint_value(
    name = "gcc",
    constraint_setting = ":cc_compiler",
)

constraint_value(
    name = "msvc",
    constraint_setting = ":cc_compiler",
)

constraint_value(
    name = "clang-cl",
    constraint_setting = ":cc_compiler",
)

constraint_value(
    name = "mingw",
    constraint_setting = ":cc_compiler",
)

constraint_value(
    name = "msys",
    constraint_setting = ":cc_compiler",
)

cc_toolchain_alias(name = "current_cc_toolchain")

cc_host_toolchain_alias(name = "current_cc_host_toolchain")

cc_libc_top_alias(name = "current_libc_top")

cc_library(
    name = "malloc",
)

filegroup(
    name = "grep-includes",
    srcs = ["grep-includes.sh"],
)

filegroup(
    name = "empty",
    srcs = [],
)

config_setting(
    name = "disable_static_cc_toolchains",
    values = {
        "incompatible_disable_static_cc_toolchains": "true",
    },
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# "cc-compiler-${CPU}" in this BUILD file, where CPU is the target CPU
# specified in --cpu.
#
# This file group should include
#   * all cc_toolchain targets supported
#   * all file groups that said cc_toolchain might refer to
alias(
    name = "toolchain",
    actual = "@local_config_cc//:toolchain",
)

# Hardcoded toolchain, legacy behaviour.
cc_toolchain_suite(
    name = "default-toolchain",
    toolchains = {
        "armeabi-v7a|compiler": ":cc-compiler-armeabi-v7a",
        "darwin|compiler": ":cc-compiler-darwin",
        "freebsd|compiler": ":cc-compiler-freebsd",
        "local|compiler": ":cc-compiler-local",
        "x64_windows|compiler": ":cc-compiler-x64_windows",
        "x64_windows_msvc|compiler": ":cc-compiler-x64_windows_msvc",
        "s390x|compiler": ":cc-compiler-s390x",
        "ppc|compiler": ":cc-compiler-ppc",
        "k8": ":cc-compiler-local",
        "piii": ":cc-compiler-local",
        "arm": ":cc-compiler-local",
        "aarch64": ":cc-compiler-local",
        "s390x": ":cc-compiler-local",
        "ppc": ":cc-compiler-local",
        "ppc64": ":cc-compiler-local",
        "darwin": ":cc-compiler-darwin",
        "freebsd": ":cc-compiler-freebsd",
        "armeabi-v7a": ":cc-compiler-armeabi-v7a",
        "x64_windows": ":cc-compiler-x64_windows",
        "x64_windows_msvc": ":cc-compiler-x64_windows_msvc",
    },
)

cc_toolchain(
    name = "cc-compiler-local",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_linux",
    toolchain_identifier = "local_linux",
)

cc_toolchain_config(
    name = "local_linux",
    compiler = "compiler",
    cpu = "local",
    disable_static_cc_toolchains = select({
        ":disable_static_cc_toolchains": True,
        "//conditions:default": False,
    }),
)

toolchain(
    name = "cc-toolchain-local",
    exec_compatible_with = [
    ],
    target_compatible_with = [
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-ppc",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_linux",
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-ppc",
    exec_compatible_with = [
        "@platforms//cpu:ppc",
    ],
    target_compatible_with = [
        "@platforms//cpu:ppc",
    ],
    toolchain = ":cc-compiler-ppc",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-s390x",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_linux",
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-s390x",
    exec_compatible_with = [
        "@platforms//cpu:s390x",
    ],
    target_compatible_with = [
        "@platforms//cpu:s390x",
    ],
    toolchain = ":cc-compiler-s390x",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_linux",
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-armeabi-v7a",
    exec_compatible_with = [
    ],
    target_compatible_with = [
        "@platforms//cpu:arm",
        "@platforms//os:android",
    ],
    toolchain = ":cc-compiler-armeabi-v7a",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-k8",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":local_linux",
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-k8",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    toolchain = ":cc-compiler-k8",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-darwin",
    all_files = ":osx_wrapper",
    ar_files = ":empty",
    as_files = ":osx_wrapper",
    compiler_files = ":osx_wrapper",
    dwp_files = ":empty",
    linker_files = ":osx_wrapper",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":local_darwin",
    toolchain_identifier = "local_darwin",
)

cc_toolchain_config(
    name = "local_darwin",
    compiler = "compiler",
    cpu = "darwin",
    disable_static_cc_toolchains = select({
        ":disable_static_cc_toolchains": True,
        "//conditions:default": False,
    }),
)

toolchain(
    name = "cc-toolchain-darwin",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:osx",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:osx",
    ],
    toolchain = ":cc-compiler-darwin",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-freebsd",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":local_freebsd",
    toolchain_identifier = "local_freebsd",
)

cc_toolchain_config(
    name = "local_freebsd",
    compiler = "compiler",
    cpu = "freebsd",
    disable_static_cc_toolchains = select({
        ":disable_static_cc_toolchains": True,
        "//conditions:default": False,
    }),
)

toolchain(
    name = "cc-toolchain-freebsd",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:freebsd",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:freebsd",
    ],
    toolchain = ":cc-compiler-freebsd",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":local_windows_msys64",
    toolchain_identifier = "local_windows_msys64",
)

cc_toolchain_config(
    name = "local_windows_msys64",
    compiler = "windows_msys64",
    cpu = "x64_windows",
    disable_static_cc_toolchains = select({
        ":disable_static_cc_toolchains": True,
        "//conditions:default": False,
    }),
)

toolchain(
    name = "cc-toolchain-x64_windows",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows_msvc",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_config = ":vc_14_0_x64",
    toolchain_identifier = "vc_14_0_x64",
)

cc_toolchain_config(
    name = "vc_14_0_x64",
    compiler = "cl",
    cpu = "x64_windows_msvc",
    disable_static_cc_toolchains = select({
        ":disable_static_cc_toolchains": True,
        "//conditions:default": False,
    }),
)

toolchain(
    name = "cc-toolchain-x64_windows_msvc",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
        ":msvc",
    ],
    target_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:windows",
    ],
    toolchain = ":cc-compiler-x64_windows_msvc",
    toolchain_type = ":toolchain_type",
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]) + [
        "//tools/cpp/runfiles:srcs",
    ],
)

filegroup(
    name = "embedded_tools",
    srcs = glob(["**"]) + [
        "//tools/cpp/runfiles:embedded_tools",
    ],
)

filegroup(
    name = "interface_library_builder",
    srcs = ["build_interface_so"],
)

filegroup(
    name = "osx_wrapper",
    srcs = ["osx_cc_wrapper.sh"],
)

filegroup(
    name = "link_dynamic_library",
    srcs = ["link_dynamic_library.sh"],
)

filegroup(
    name = "lib_cc_configure",
    srcs = ["lib_cc_configure.bzl"],
)

filegroup(
    name = "bzl_srcs",
    srcs = glob(["*.bzl"]),
    visibility = ["//tools:__pkg__"],
)

load(":compiler_flag.bzl", "compiler_flag")

compiler_flag(name = "compiler")

# Target that can provide the CC_FLAGS variable based on the current
# cc_toolchain.
load("@bazel_tools//tools/cpp:cc_flags_supplier.bzl", "cc_flags_supplier")

cc_flags_supplier(name = "cc_flags")
