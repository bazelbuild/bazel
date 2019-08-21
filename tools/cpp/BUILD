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

alias(
    name = "clang",
    actual = "@rules_cc//cc/private/toolchain:clang",
)

alias(
    name = "gcc",
    actual = "@rules_cc//cc/private/toolchain:gcc",
)

alias(
    name = "msvc",
    actual = "@rules_cc//cc/private/toolchain:msvc",
)

alias(
    name = "clang-cl",
    actual = "@rules_cc//cc/private/toolchain:clang-cl",
)

alias(
    name = "mingw",
    actual = "@rules_cc//cc/private/toolchain:mingw",
)

alias(
    name = "msys",
    actual = "@rules_cc//cc/private/toolchain:msys",
)

alias(
    name = "current_cc_toolchain",
    actual = "@rules_cc//cc/private/toolchain:current_cc_toolchain",
)

alias(
    name = "current_cc_host_toolchain",
    actual = "@rules_cc//cc/private/toolchain:current_cc_host_toolchain",
)

alias(
    name = "current_libc_top",
    actual = "@rules_cc//cc/private/toolchain:current_libc_top",
)

alias(
    name = "malloc",
    actual = "@rules_cc//cc/private/toolchain:malloc",
)

alias(
    name = "grep-includes",
    actual = "@rules_cc//cc/private/toolchain:grep-includes",
)

alias(
    name = "empty",
    actual = "@rules_cc//cc/private/toolchain:empty",
)

alias(
    name = "disable_static_cc_toolchains",
    actual = "@rules_cc//cc/private/toolchain:disable_static_cc_toolchains",
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
    actual = "@rules_cc//cc/private/toolchain:toolchain",
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
        "@rules_cc//cc/private/toolchain:embedded_tools",
        "//tools/cpp/runfiles:embedded_tools",
    ],
)

filegroup(
    name = "embedded_tools_toolchain_type",
    srcs = ["BUILD"],
)

alias(
    name = "interface_library_builder",
    actual = "@rules_cc//cc/private/toolchain:interface_library_builder",
)

alias(
    name = "osx_wrapper",
    actual = "@rules_cc//cc/private/toolchain:osx_wrapper",
)

alias(
    name = "link_dynamic_library",
    actual = "@rules_cc//cc/private/toolchain:link_dynamic_library",
)

alias(
    name = "lib_cc_configure",
    actual = "@rules_cc//cc/private/toolchain:lib_cc_configure",
)

alias(
    name = "compiler",
    actual = "@rules_cc//cc/private/toolchain:compiler",
)

# Target that can provide the CC_FLAGS variable based on the current
# cc_toolchain.
alias(
    name = "cc_flags",
    actual = "@rules_cc//cc/private/toolchain:cc_flags",
)
