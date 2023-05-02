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

licenses(["notice"])

# The toolchain type used to distinguish cc toolchains.
toolchain_type(name = "toolchain_type")

# The toolchain_type used for cc test runners.
toolchain_type(name = "test_runner_toolchain_type")

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

cc_toolchain_alias(
    name = "optional_current_cc_toolchain",
    mandatory = False,
)

cc_libc_top_alias(name = "current_libc_top")

cc_library(
    name = "empty_lib",
    tags = ["__DONT_DEPEND_ON_DEF_PARSER__"],
)

# Label flag for extra libraries to be linked into every binary.
# TODO(bazel-team): Support passing flag multiple times to build a list.
label_flag(
    name = "link_extra_libs",
    build_setting_default = ":empty_lib",
)

# The final extra library to be linked into every binary target. This collects
# the above flag, but may also include more libraries depending on config.
cc_library(
    name = "link_extra_lib",
    tags = ["__DONT_DEPEND_ON_DEF_PARSER__"],
    deps = [
        ":link_extra_libs",
    ],
)

cc_library(
    name = "malloc",
    tags = ["__DONT_DEPEND_ON_DEF_PARSER__"],
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

# The following is the default value for --xcode_version_config.
alias(
    name = "host_xcodes",
    actual = "@local_config_xcode//:host_xcodes",
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

cc_library(
    name = "build_info_translation_helper",
    srcs = [":build_info_translation_helper.cc"],
    hdrs = [":build_info_translation_helper.h"],
    deps = [
        "@com_google_absl//absl/status",
        "@com_google_absl//absl/strings"],
)

cc_library(
    name = "bazel_build_info_translation_helper",
    srcs = [":bazel_build_info_translation_helper.cc"],
    hdrs = [":bazel_build_info_translation_helper.h"],
    deps = [":build_info_translation_helper"],
)

load(":compiler_flag.bzl", "compiler_flag")

compiler_flag(name = "compiler")

# Target that can provide the CC_FLAGS variable based on the current
# cc_toolchain.
load("@bazel_tools//tools/cpp:cc_flags_supplier.bzl", "cc_flags_supplier")

cc_flags_supplier(name = "cc_flags")
