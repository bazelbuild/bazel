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
"""Rules for configuring the C++ toolchain (experimental)."""

load("@bazel_tools//tools/cpp:windows_cc_configure.bzl", "configure_windows_toolchain")
load("@bazel_tools//tools/cpp:osx_cc_configure.bzl", "configure_osx_toolchain")
load("@bazel_tools//tools/cpp:unix_cc_configure.bzl", "configure_unix_toolchain")
load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "get_cpu_value",
    "resolve_labels",
)

def cc_autoconf_impl(repository_ctx, overriden_tools = dict()):
    paths = resolve_labels(repository_ctx, [
        "@bazel_tools//tools/cpp:BUILD.static.freebsd",
        "@bazel_tools//tools/cpp:cc_toolchain_config.bzl",
        "@bazel_tools//tools/cpp:dummy_toolchain.bzl",
    ])

    repository_ctx.symlink(
        paths["@bazel_tools//tools/cpp:dummy_toolchain.bzl"],
        "dummy_toolchain.bzl",
    )

    env = repository_ctx.os.environ
    cpu_value = get_cpu_value(repository_ctx)
    if "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN" in env and env["BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN"] == "1":
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.symlink(Label("@bazel_tools//tools/cpp:BUILD.empty"), "BUILD")
    elif cpu_value == "freebsd":
        # This is defaulting to the static crosstool, we should eventually
        # autoconfigure this platform too.  Theorically, FreeBSD should be
        # straightforward to add but we cannot run it in a docker container so
        # skipping until we have proper tests for FreeBSD.
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:BUILD.static.freebsd"], "BUILD")
    elif cpu_value == "x64_windows":
        # TODO(ibiryukov): overriden_tools are only supported in configure_unix_toolchain.
        # We might want to add that to Windows too(at least for msys toolchain).
        configure_windows_toolchain(repository_ctx)
    elif (cpu_value == "darwin" and
          ("BAZEL_USE_CPP_ONLY_TOOLCHAIN" not in env or env["BAZEL_USE_CPP_ONLY_TOOLCHAIN"] != "1")):
        configure_osx_toolchain(repository_ctx, overriden_tools)
    else:
        configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools)

cc_autoconf = repository_rule(
    environ = [
        "ABI_LIBC_VERSION",
        "ABI_VERSION",
        "BAZEL_COMPILER",
        "BAZEL_HOST_SYSTEM",
        "BAZEL_CXXOPTS",
        "BAZEL_LINKOPTS",
        "BAZEL_PYTHON",
        "BAZEL_SH",
        "BAZEL_TARGET_CPU",
        "BAZEL_TARGET_LIBC",
        "BAZEL_TARGET_SYSTEM",
        "BAZEL_USE_CPP_ONLY_TOOLCHAIN",
        "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN",
        "BAZEL_USE_LLVM_NATIVE_COVERAGE",
        "BAZEL_VC",
        "BAZEL_VC_TOOL",
        "BAZEL_VS",
        "BAZEL_LLVM",
        "USE_CLANG_CL",
        "CC",
        "CC_CONFIGURE_DEBUG",
        "CC_TOOLCHAIN_NAME",
        "CPLUS_INCLUDE_PATH",
        "GCOV",
        "HOMEBREW_RUBY_PATH",
        "SYSTEMROOT",
        "VS90COMNTOOLS",
        "VS100COMNTOOLS",
        "VS110COMNTOOLS",
        "VS120COMNTOOLS",
        "VS140COMNTOOLS",
    ],
    implementation = cc_autoconf_impl,
)

def cc_configure():
    """A C++ configuration rules that generate the crosstool file."""
    cc_autoconf(name = "local_config_cc")
    native.bind(name = "cc_toolchain", actual = "@local_config_cc//:toolchain")
    native.register_toolchains(
        # Use register_toolchain's target pattern expansion to register all toolchains in the package.
        "@local_config_cc//:all",
    )
