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
        "armeabi-v7a|compiler": ":cc-compiler-armeabi-v7a",
        "ios_x86_64|compiler": ":cc-compiler-ios_x86_64",
    },
)

cc_toolchain(
    name = "cc-compiler-%{name}",
    all_files = ":compiler_deps",
    compiler_files = ":compiler_deps",
    cpu = "%{name}",
    dwp_files = ":empty",
    dynamic_runtime_libs = [":empty"],
    linker_files = ":compiler_deps",
    objcopy_files = ":empty",
    static_runtime_libs = [":empty"],
    strip_files = ":empty",
    supports_param_files = %{supports_param_files},
)


# Android tooling requires a default toolchain for the armeabi-v7a cpu.
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
)

# ios crosstool configuration requires a default toolchain for the
# ios_x86_64 cpu.
cc_toolchain(
    name = "cc-compiler-ios_x86_64",
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
)

cc_toolchain_type(name = "toolchain_type")

# A dummy toolchain is necessary to satisfy toolchain resolution until platforms
# are used in c++ by default.
# TODO(b/64754003): Remove once platforms are used in c++ by default.
toolchain(
    name = "dummy_cc_toolchain",
    toolchain = "dummy_cc_toolchain_impl",
    toolchain_type = ":toolchain_type",
)

toolchain(
    name = "dummy_cc_toolchain_type",
    toolchain = "dummy_cc_toolchain_impl",
    toolchain_type = ":toolchain_type",
)

load(":dummy_toolchain.bzl", "dummy_toolchain")

dummy_toolchain(name = "dummy_cc_toolchain_impl")
