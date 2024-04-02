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

load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "get_cpu_value",
    "resolve_labels",
)
load("@bazel_tools//tools/cpp:unix_cc_configure.bzl", "configure_unix_toolchain")
load("@bazel_tools//tools/cpp:windows_cc_configure.bzl", "configure_windows_toolchain")

def cc_autoconf_toolchains_impl(repository_ctx):
    """Generate BUILD file with 'toolchain' targets for the local host C++ toolchain.

    Args:
      repository_ctx: repository context
    """
    env = repository_ctx.os.environ

    # Should we try to find C++ toolchain at all? If not, we don't have to generate toolchains for C++ at all.
    should_detect_cpp_toolchain = "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN" not in env or env["BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN"] != "1"

    if should_detect_cpp_toolchain:
        paths = resolve_labels(repository_ctx, [
            "@bazel_tools//tools/cpp:BUILD.toolchains.tpl",
        ])
        repository_ctx.template(
            "BUILD",
            paths["@bazel_tools//tools/cpp:BUILD.toolchains.tpl"],
            {"%{name}": get_cpu_value(repository_ctx)},
        )
    else:
        repository_ctx.file("BUILD", "# C++ toolchain autoconfiguration was disabled by BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN env variable.")

cc_autoconf_toolchains = repository_rule(
    environ = [
        "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN",
    ],
    implementation = cc_autoconf_toolchains_impl,
    configure = True,
)

def cc_autoconf_impl(repository_ctx, overriden_tools = dict()):
    """Generate BUILD file with 'cc_toolchain' targets for the local host C++ toolchain.

    Args:
       repository_ctx: repository context
       overriden_tools: dict of tool paths to use instead of autoconfigured tools
    """

    env = repository_ctx.os.environ
    cpu_value = get_cpu_value(repository_ctx)
    if "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN" in env and env["BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN"] == "1":
        paths = resolve_labels(repository_ctx, [
            "@bazel_tools//tools/cpp:BUILD.empty.tpl",
            "@bazel_tools//tools/cpp:empty_cc_toolchain_config.bzl",
        ])
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:empty_cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.template("BUILD", paths["@bazel_tools//tools/cpp:BUILD.empty.tpl"], {
            "%{cpu}": get_cpu_value(repository_ctx),
        })
    elif cpu_value == "freebsd" or cpu_value == "openbsd":
        paths = resolve_labels(repository_ctx, [
            "@bazel_tools//tools/cpp:BUILD.static.bsd",
            "@bazel_tools//tools/cpp:bsd_cc_toolchain_config.bzl",
        ])

        # This is defaulting to a static crosstool. We should eventually
        # autoconfigure this platform too. Theoretically, FreeBSD and OpenBSD
        # should be straightforward to add but we cannot run them in a Docker
        # container so skipping until we have proper tests for these platforms.
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:bsd_cc_toolchain_config.bzl"], "cc_toolchain_config.bzl")
        repository_ctx.symlink(paths["@bazel_tools//tools/cpp:BUILD.static.bsd"], "BUILD")
    elif cpu_value in ["x64_windows", "arm64_windows"]:
        # TODO(ibiryukov): overriden_tools are only supported in configure_unix_toolchain.
        # We might want to add that to Windows too(at least for msys toolchain).
        configure_windows_toolchain(repository_ctx)
    else:
        configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools)

MSVC_ENVVARS = [
    "BAZEL_VC",
    "BAZEL_VC_FULL_VERSION",
    "BAZEL_VS",
    "BAZEL_WINSDK_FULL_VERSION",
    "VS90COMNTOOLS",
    "VS100COMNTOOLS",
    "VS110COMNTOOLS",
    "VS120COMNTOOLS",
    "VS140COMNTOOLS",
    "VS150COMNTOOLS",
    "VS160COMNTOOLS",
    "TMP",
    "TEMP",
]

cc_autoconf = repository_rule(
    environ = [
        "ABI_LIBC_VERSION",
        "ABI_VERSION",
        "BAZEL_COMPILER",
        "BAZEL_HOST_SYSTEM",
        "BAZEL_CONLYOPTS",
        "BAZEL_CXXOPTS",
        "BAZEL_LINKOPTS",
        "BAZEL_LINKLIBS",
        "BAZEL_LLVM_COV",
        "BAZEL_LLVM_PROFDATA",
        "BAZEL_PYTHON",
        "BAZEL_SH",
        "BAZEL_TARGET_CPU",
        "BAZEL_TARGET_LIBC",
        "BAZEL_TARGET_SYSTEM",
        "BAZEL_DO_NOT_DETECT_CPP_TOOLCHAIN",
        "BAZEL_USE_LLVM_NATIVE_COVERAGE",
        "BAZEL_LLVM",
        "BAZEL_IGNORE_SYSTEM_HEADERS_VERSIONS",
        "USE_CLANG_CL",
        "CC",
        "CC_CONFIGURE_DEBUG",
        "CC_TOOLCHAIN_NAME",
        "CPLUS_INCLUDE_PATH",
        "DEVELOPER_DIR",
        "GCOV",
        "LIBTOOL",
        "HOMEBREW_RUBY_PATH",
        "SYSTEMROOT",
        "USER",
    ] + MSVC_ENVVARS,
    implementation = cc_autoconf_impl,
    configure = True,
)

def cc_configure():
    """A C++ configuration rules that generate the crosstool file."""
    cc_autoconf_toolchains(name = "local_config_cc_toolchains")
    cc_autoconf(name = "local_config_cc")
    native.bind(name = "cc_toolchain", actual = "@local_config_cc//:toolchain")
    native.register_toolchains(
        # Use register_toolchain's target pattern expansion to register all toolchains in the package.
        "@local_config_cc_toolchains//:all",
    )

def _cc_configure_extension_impl(ctx):
    cc_autoconf_toolchains(name = "local_config_cc_toolchains")
    cc_autoconf(name = "local_config_cc")

cc_configure_extension = module_extension(implementation = _cc_configure_extension_impl)
