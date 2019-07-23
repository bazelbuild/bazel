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

# Hardcoded toolchain, legacy behaviour.
alias(
    name = "default-toolchain",
    actual = "@rules_cc//cc/private/toolchain:default-toolchain",
)

alias(
    name = "cc-compiler-local",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-local",
)

alias(
    name = "local_linux",
    actual = "@rules_cc//cc/private/toolchain:local_linux",
)

alias(
    name = "cc-toolchain-local",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-local",
)

alias(
    name = "cc-compiler-ppc",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-ppc",
)

alias(
    name = "cc-toolchain-ppc",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-ppc",
)

alias(
    name = "cc-compiler-s390x",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-s390x",
)

alias(
    name = "cc-toolchain-s390x",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-s390x",
)

alias(
    name = "cc-compiler-armeabi-v7a",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-armeabi-v7a",
)

alias(
    name = "cc-toolchain-armeabi-v7a",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-armeabi-v7a",
)

alias(
    name = "cc-compiler-k8",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-k8",
)

alias(
    name = "cc-toolchain-k8",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-k8",
)

alias(
    name = "cc-compiler-darwin",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-darwin",
)

alias(
    name = "local_darwin",
    actual = "@rules_cc//cc/private/toolchain:local_darwin",
)

alias(
    name = "cc-toolchain-darwin",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-darwin",
)

alias(
    name = "cc-compiler-freebsd",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-freebsd",
)

alias(
    name = "local_freebsd",
    actual = "@rules_cc//cc/private/toolchain:local_freebsd",
)

alias(
    name = "cc-toolchain-freebsd",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-freebsd",
)

alias(
    name = "cc-compiler-x64_windows",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-x64_windows",
)

alias(
    name = "local_windows_msys64",
    actual = "@rules_cc//cc/private/toolchain:local_windows_msys64",
)

alias(
    name = "cc-toolchain-x64_windows",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-x64_windows",
)

alias(
    name = "cc-compiler-x64_windows_msvc",
    actual = "@rules_cc//cc/private/toolchain:cc-compiler-x64_windows_msvc",
)

alias(
    name = "vc_14_0_x64",
    actual = "@rules_cc//cc/private/toolchain:vc_14_0_x64",
)

alias(
    name = "cc-toolchain-x64_windows_msvc",
    actual = "@rules_cc//cc/private/toolchain:cc-toolchain-x64_windows_msvc",
)

alias(
    name = "srcs",
    actual = "@rules_cc//cc/private/toolchain:srcs",
)

alias(
    name = "embedded_tools",
    actual = "@rules_cc//cc/private/toolchain:embedded_tools",
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
