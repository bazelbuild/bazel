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

cc_library(
    name = "stl",
)

filegroup(
    name = "grep-includes",
    srcs = ["grep-includes.sh"],
)

filegroup(
    name = "empty",
    srcs = [],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# "cc-compiler-${CPU}" in this BUILD file, where CPU is the target CPU
# specified in --cpu.
#
# This file group should include
#   * all cc_toolchain targets supported
#   * all file groups that said cc_toolchain might refer to,
# including the default_grte_top setting in the CROSSTOOL
# protobuf.
alias(
    name = "toolchain",
    actual = "//external:cc_toolchain",
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
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-local",
    exec_compatible_with = [
        "@bazel_tools//platforms:autoconfigured",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:autoconfigured",
    ],
    toolchain = ":cc-compiler-local",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-ppc",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "ppc",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-ppc",
    exec_compatible_with = [
        "@bazel_tools//platforms:ppc",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:ppc",
    ],
    toolchain = ":cc-compiler-ppc",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-armeabi-v7a",
    exec_compatible_with = [
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:arm",
        "@bazel_tools//platforms:android",
    ],
    toolchain = ":cc-compiler-armeabi-v7a",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-k8",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "local_linux",
)

toolchain(
    name = "cc-toolchain-k8",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:linux",
    ],
    toolchain = ":cc-compiler-k8",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-darwin",
    all_files = ":osx_wrapper",
    compiler_files = ":osx_wrapper",
    cpu = "darwin",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":osx_wrapper",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_identifier = "local_darwin",
)

toolchain(
    name = "cc-toolchain-darwin",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:osx",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:osx",
    ],
    toolchain = ":cc-compiler-darwin",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-freebsd",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_identifier = "local_freebsd",
)

toolchain(
    name = "cc-toolchain-freebsd",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:freebsd",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:freebsd",
    ],
    toolchain = ":cc-compiler-freebsd",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows",
    all_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_identifier = "local_windows_msys64",
)

toolchain(
    name = "cc-toolchain-x64_windows",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    toolchain = ":cc-compiler-x64_windows",
    toolchain_type = ":toolchain_type",
)

cc_toolchain(
    name = "cc-compiler-x64_windows_msvc",
    all_files = ":every-file-x64_windows",
    compiler_files = ":compile-x64_windows",
    cpu = "x64_windows",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":empty",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = 1,
    toolchain_identifier = "vc_14_0_x64",
)

toolchain(
    name = "cc-toolchain-x64_windows_msvc",
    exec_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
        ":msvc",
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:x86_64",
        "@bazel_tools//platforms:windows",
    ],
    toolchain = ":cc-compiler-x64_windows_msvc",
    toolchain_type = ":toolchain_type",
)

filegroup(
    name = "every-file-x64_windows",
    srcs = [
        ":compile-x64_windows",
    ],
)

filegroup(
    name = "compile-x64_windows",
    srcs = glob([
        "wrapper/bin/msvc_*",
        "wrapper/bin/pydir/msvc*",
    ]),
)

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
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
    name = "crosstool_utils",
    srcs = ["crosstool_utils.bzl"],
)

filegroup(
    name = "crosstool_lib",
    srcs = ["crosstool_lib.bzl"],
)

load(":compiler_flag.bzl", "compiler_flag")

compiler_flag(name = "compiler")
