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

load(
    "@bazel_tools//tools/osx:xcode_configure.bzl",
    "OSX_EXECUTE_TIMEOUT",
    "run_xcode_locator",
)
load(
    "@bazel_tools//tools/cpp:lib_cc_configure.bzl",
    "escape_string",
    "resolve_labels",
)
load(
    "@bazel_tools//tools/cpp:unix_cc_configure.bzl",
    "configure_unix_toolchain",
    "get_env",
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

    # Assume that everything is managed by Xcode / toolchain installations
    include_dirs = [
        "/Applications/",
        "/Library/",
    ]

    user = repository_ctx.os.environ.get("USER")
    if user:
        include_dirs.append("/Users/{}/Library/".format(user))

    # Include extra Xcode paths in case they're installed on other volumes
    for toolchain in xcode_toolchains:
        include_dirs.append(escape_string(toolchain.developer_dir))

    return include_dirs

# TODO: Remove once Xcode 12 is the minimum supported version
def _compile_cc_file_single_arch(repository_ctx, src_name, out_name, timeout):
    env = repository_ctx.os.environ
    xcrun_result = repository_ctx.execute([
        "env",
        "-i",
        "DEVELOPER_DIR={}".format(env.get("DEVELOPER_DIR", default = "")),
        "xcrun",
        "--sdk",
        "macosx",
        "clang",
        "-mmacosx-version-min=10.13",
        "-std=c++11",
        "-lc++",
        "-O3",
        "-o",
        out_name,
        src_name,
    ], timeout)
    if (xcrun_result.return_code != 0):
        error_msg = (
            "return code {code}, stderr: {err}, stdout: {out}"
        ).format(
            code = xcrun_result.return_code,
            err = xcrun_result.stderr,
            out = xcrun_result.stdout,
        )
        fail(out_name + " failed to generate. Please file an issue at " +
             "https://github.com/bazelbuild/bazel/issues with the following:\n" +
             error_msg)

def _compile_cc_file(repository_ctx, src_name, out_name, timeout):
    env = repository_ctx.os.environ
    xcrun_result = repository_ctx.execute([
        "env",
        "-i",
        "DEVELOPER_DIR={}".format(env.get("DEVELOPER_DIR", default = "")),
        "xcrun",
        "--sdk",
        "macosx",
        "clang",
        "-mmacosx-version-min=10.13",
        "-std=c++11",
        "-lc++",
        "-arch",
        "arm64",
        "-arch",
        "x86_64",
        "-O3",
        "-o",
        out_name,
        src_name,
    ], timeout)

    if xcrun_result.return_code == 0:
        xcrun_result = repository_ctx.execute([
            "env",
            "-i",
            "codesign",
            "--identifier",  # Required to be reproducible across archs
            out_name,
            "--force",
            "--sign",
            "-",
            out_name,
        ], timeout)
        if xcrun_result.return_code != 0:
            error_msg = (
                "codesign return code {code}, stderr: {err}, stdout: {out}"
            ).format(
                code = xcrun_result.return_code,
                err = xcrun_result.stderr,
                out = xcrun_result.stdout,
            )
            fail(out_name + " failed to generate. Please file an issue at " +
                 "https://github.com/bazelbuild/bazel/issues with the following:\n" +
                 error_msg)
    else:
        _compile_cc_file_single_arch(repository_ctx, src_name, out_name, timeout)

def configure_osx_toolchain(repository_ctx, cpu_value, overriden_tools):
    """Configure C++ toolchain on macOS.

    Args:
      repository_ctx: The repository context.
      overriden_tools: dictionary of overridden tools.
    """
    paths = resolve_labels(repository_ctx, [
        "@bazel_tools//tools/cpp:armeabi_cc_toolchain_config.bzl",
        "@bazel_tools//tools/cpp:osx_cc_wrapper.sh.tpl",
        "@bazel_tools//tools/objc:libtool.sh",
        "@bazel_tools//tools/objc:libtool_check_unique.cc",
        "@bazel_tools//tools/objc:make_hashed_objlist.py",
        "@bazel_tools//tools/objc:xcrunwrapper.sh",
        "@bazel_tools//tools/osx/crosstool:BUILD.tpl",
        "@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl",
        "@bazel_tools//tools/osx/crosstool:wrapped_clang.cc",
        "@bazel_tools//tools/osx:xcode_locator.m",
    ])

    env = repository_ctx.os.environ
    should_use_xcode = "BAZEL_USE_XCODE_TOOLCHAIN" in env and env["BAZEL_USE_XCODE_TOOLCHAIN"] == "1"
    if "BAZEL_OSX_EXECUTE_TIMEOUT" in env:
        timeout = int(env["BAZEL_OSX_EXECUTE_TIMEOUT"])
    else:
        timeout = OSX_EXECUTE_TIMEOUT
    xcode_toolchains = []

    # Make the following logic in sync with //tools/cpp:cc_configure.bzl#cc_autoconf_toolchains_impl
    (xcode_toolchains, xcodeloc_err) = run_xcode_locator(
        repository_ctx,
        paths["@bazel_tools//tools/osx:xcode_locator.m"],
    )
    if should_use_xcode and not xcode_toolchains:
        fail("BAZEL_USE_XCODE_TOOLCHAIN is set to 1 but Bazel couldn't find Xcode installed on the " +
             "system. Verify that 'xcode-select -p' is correct.")
    if xcode_toolchains:
        # For Xcode toolchains, there's no reason to use anything other than
        # wrapped_clang, so that we still get the Bazel Xcode placeholder
        # substitution and other behavior for actions that invoke this
        # cc_wrapper.sh script. The wrapped_clang binary is already hardcoded
        # into the Objective-C crosstool actions, anyway, so this ensures that
        # the C++ actions behave consistently.
        cc = repository_ctx.path("wrapped_clang")

        cc_path = '"$(/usr/bin/dirname "$0")"/wrapped_clang'
        repository_ctx.template(
            "cc_wrapper.sh",
            paths["@bazel_tools//tools/cpp:osx_cc_wrapper.sh.tpl"],
            {
                "%{cc}": escape_string(cc_path),
                "%{env}": escape_string(get_env(repository_ctx)),
            },
        )
        repository_ctx.symlink(
            paths["@bazel_tools//tools/cpp:armeabi_cc_toolchain_config.bzl"],
            "armeabi_cc_toolchain_config.bzl",
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
            paths["@bazel_tools//tools/osx/crosstool:cc_toolchain_config.bzl"],
            "cc_toolchain_config.bzl",
        )
        libtool_check_unique_src_path = str(repository_ctx.path(
            paths["@bazel_tools//tools/objc:libtool_check_unique.cc"],
        ))
        _compile_cc_file(
            repository_ctx,
            libtool_check_unique_src_path,
            "libtool_check_unique",
            timeout,
        )
        wrapped_clang_src_path = str(repository_ctx.path(
            paths["@bazel_tools//tools/osx/crosstool:wrapped_clang.cc"],
        ))
        _compile_cc_file(repository_ctx, wrapped_clang_src_path, "wrapped_clang", timeout)
        repository_ctx.symlink("wrapped_clang", "wrapped_clang_pp")

        tool_paths = {}
        gcov_path = repository_ctx.os.environ.get("GCOV")
        if gcov_path != None:
            if not gcov_path.startswith("/"):
                gcov_path = repository_ctx.which(gcov_path)
            tool_paths["gcov"] = gcov_path

        escaped_include_paths = _get_escaped_xcode_cxx_inc_directories(repository_ctx, cc, xcode_toolchains)
        escaped_cxx_include_directories = []
        for path in escaped_include_paths:
            escaped_cxx_include_directories.append(("            \"%s\"," % path))
        if xcodeloc_err:
            escaped_cxx_include_directories.append("            # Error: " + xcodeloc_err)
        repository_ctx.template(
            "BUILD",
            paths["@bazel_tools//tools/osx/crosstool:BUILD.tpl"],
            {
                "%{cxx_builtin_include_directories}": "\n".join(escaped_cxx_include_directories),
                "%{tool_paths_overrides}": ",\n            ".join(
                    ['"%s": "%s"' % (k, v) for k, v in tool_paths.items()],
                ),
            },
        )
    else:
        configure_unix_toolchain(repository_ctx, cpu_value, overriden_tools = overriden_tools)
