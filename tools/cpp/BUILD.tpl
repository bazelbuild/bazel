# Copyright 2016 The Bazel Authors. All rights reserved.
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

# This becomes the BUILD file for @local_config_cc// under non-FreeBSD unixes.

package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

cc_library(
    name = "malloc",
)

cc_library(
    name = "stl",
)

filegroup(
    name = "empty",
    srcs = [],
)

filegroup(
    name = "cc_wrapper",
    srcs = ["cc_wrapper.sh"],
)

filegroup(
    name = "compiler_deps",
    srcs = glob(["extra_tools/**"]) + ["%{cc_compiler_deps}"],
)

# This is the entry point for --crosstool_top.  Toolchains are found
# by lopping off the name of --crosstool_top and searching for
# the "${CPU}" entry in the toolchains attribute.
cc_toolchain_suite(
    name = "toolchain",
    toolchains = {
        "%{name}|%{compiler}": ":cc-compiler-%{name}",
        "%{name}": ":cc-compiler-%{name}",
        "armeabi-v7a|compiler": ":cc-compiler-armeabi-v7a",
        "armeabi-v7a": ":cc-compiler-armeabi-v7a",
    },
)

cc_toolchain(
    name = "cc-compiler-%{name}",
    toolchain_identifier = "%{cc_toolchain_identifier}",
    all_files = ":compiler_deps",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":compiler_deps",
    cpu = "%{name}",
    dwp_files = ":empty",
    linker_files = ":compiler_deps",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = %{supports_param_files},
)

toolchain(
    name = "cc-toolchain-%{name}",
    exec_compatible_with = [
        # TODO(katre): add autodiscovered constraints for host CPU and OS.
    ],
    target_compatible_with = [
        # TODO(katre): add autodiscovered constraints for host CPU and OS.
    ],
    toolchain = ":cc-compiler-%{name}",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# Android tooling requires a default toolchain for the armeabi-v7a cpu.
cc_toolchain(
    name = "cc-compiler-armeabi-v7a",
    toolchain_identifier = "stub_armeabi-v7a",
    all_files = ":empty",
    ar_files = ":empty",
    as_files = ":empty",
    compiler_files = ":empty",
    cpu = "local",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 1,
)

toolchain(
    name = "cc-toolchain-armeabi-v7a",
    exec_compatible_with = [
        # TODO(katre): add autodiscovered constraints for host CPU and OS.
    ],
    target_compatible_with = [
        "@bazel_tools//platforms:arm",
        "@bazel_tools//platforms:android",
    ],
    toolchain = ":cc-compiler-armabi-v7a",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

# Target that can provide the CC_FLAGS variable based on the current
# cc_toolchain.
load("@bazel_tools//tools/cpp:cc_flags_supplier.bzl", "cc_flags_supplier")

cc_flags_supplier(name = "cc_flags")
