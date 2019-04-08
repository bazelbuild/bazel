# pylint: disable=g-bad-file-header
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
"""Configuring the C++ toolchain on macOS."""

load("@bazel_tools//tools/osx:xcode_configure.bzl", "run_xcode_locator")
load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "escape_string",
    "resolve_labels",
)
load(
    "@bazel_tools//tools/cpp:unix_cc_configure.bzl",
    "configure_unix_toolchain",
    "find_cc",
    "get_env",
    "get_escaped_cxx_inc_directories",
)

def _get_escaped_xcode_cxx_inc_directories(repository_ctx, cc, xcode_toolchains):
    """Compute the list of default C++ include paths on Xcode-enabled darwin.

    Args:
      repository_ctx: The repository context.
      cc: The default C++ compiler on the local system.
      xcode_toolchains: A list containing the xcode toolchains available
    Returns:
      include_paths: A list of builtin include paths.
    """

    # TODO(cparsons): Falling back to the default C++ compiler builtin include
    # paths shouldn't be unnecessary once all actions are using xcrun.
    include_dirs = get_escaped_cxx_inc_directories(repository_ctx, cc, "-xc++")
    for toolchain in xcode_toolchains:
        include_dirs.append(escape_string(toolchain.developer_dir))

    # Assume that all paths that point to /Applications/ are built in include paths
    include_dirs.append("/Applications/")
    return include_dirs

def configure_osx_toolchain(repository_ctx, overriden_tools):
    """Configure C++ toolchain on macOS."""
    paths = resolve_labels(repository_ctx, [
        "@bazel_tools//tools/cpp:osx_cc_wrapper.sh.tpl",
        "@bazel_tools//tools/objc:libtool.sh",
        "@bazel_tools//tools/objc:make_hashed_objlist.py",
        "@bazel_tools//tools/objc:xcrunwrapper.sh",
        "@bazel_tools//tools/osx/crosstool:BUILD.tpl",
        "@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl.tpl",
        "@bazel_tools//tools/osx/crosstool:osx_archs.bzl",
        "@bazel_tools//tools/osx/crosstool:wrapped_ar.tpl",
        "@bazel_tools//tools/osx/crosstool:wrapped_clang.cc",
        "@bazel_tools//tools/osx:xcode_locator.m",
    ])

    xcode_toolchains = []
    (xcode_toolchains, xcodeloc_err) = run_xcode_locator(
        repository_ctx,
        paths["@bazel_tools//tools/osx:xcode_locator.m"],
    )
    if xcode_toolchains:
        cc = find_cc(repository_ctx, overriden_tools = {})
        repository_ctx.template(
            "cc_wrapper.sh",
            paths["@bazel_tools//tools/cpp:osx_cc_wrapper.sh.tpl"],
            {
                "%{cc}": escape_string(str(cc)),
                "%{env}": escape_string(get_env(repository_ctx)),
            },
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:xcrunwrapper.sh"],
            "xcrunwrapper.sh",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:libtool.sh"],
            "libtool",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/objc:make_hashed_objlist.py"],
            "make_hashed_objlist.py",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/osx/crosstool:wrapped_ar.tpl"],
            "wrapped_ar",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/osx/crosstool:BUILD.tpl"],
            "BUILD",
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/osx/crosstool:osx_archs.bzl"],
            "osx_archs.bzl",
        )

        wrapped_clang_src_path = str(repository_ctx.path(
            paths["@bazel_tools//tools/osx/crosstool:wrapped_clang.cc"],
        ))
        xcrun_result = repository_ctx.execute([
            "env",
            "-i",
            "xcrun",
            "clang",
            "-std=c++11",
            "-lc++",
            "-o",
            "wrapped_clang",
            wrapped_clang_src_path,
        ], 30)
        if (xcrun_result.return_code == 0):
            repository_ctx.symlink("wrapped_clang", "wrapped_clang_pp")
        else:
            error_msg = (
                "return code {code}, stderr: {err}, stdout: {out}"
            ).format(
                code = xcrun_result.return_code,
                err = xcrun_result.stderr,
                out = xcrun_result.stdout,
            )
            fail("wrapped_clang failed to generate. Please file an issue at " +
                 "https://github.com/bazelbuild/bazel/issues with the following:\n" +
                 error_msg)

        escaped_include_paths = _get_escaped_xcode_cxx_inc_directories(repository_ctx, cc, xcode_toolchains)
        escaped_cxx_include_directories = []
        for path in escaped_include_paths:
            escaped_cxx_include_directories.append(("    \"%s\"," % path))
        if xcodeloc_err:
            escaped_cxx_include_directories.append("# Error: " + xcodeloc_err + "\n")
        repository_ctx.template(
            "cc_toolchain_config.bzl",
            paths["@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl.tpl"],
            {"%{cxx_builtin_include_directories}": "\n".join(escaped_cxx_include_directories)},
        )
    else:
        configure_unix_toolchain(repository_ctx, cpu_value = "darwin", overriden_tools = overriden_tools)
