# Copyright 2021 The Bazel Authors. All rights reserved.
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

"""Starlark tests for cc_shared_library"""

load("@bazel_skylib//lib:unittest.bzl", "analysistest", "asserts", "unittest")

def _same_package_or_above(label_a, label_b):
    if label_a.workspace_name != label_b.workspace_name:
        return False
    package_a_tokenized = label_a.package.split("/")
    package_b_tokenized = label_b.package.split("/")
    if len(package_b_tokenized) < len(package_a_tokenized):
        return False

    if package_a_tokenized[0] != "":
        for i in range(len(package_a_tokenized)):
            if package_a_tokenized[i] != package_b_tokenized[i]:
                return False

    return True

def _check_if_target_under_path(value, pattern):
    if pattern.workspace_name != value.workspace_name:
        return False
    if pattern.name == "__pkg__":
        return pattern.package == value.package
    if pattern.name == "__subpackages__":
        return _same_package_or_above(pattern, value)

    return pattern.package == value.package and pattern.name == value.name

def _linking_suffix_test_impl(ctx):
    env = analysistest.begin(ctx)

    if ctx.attr.is_linux:
        target_under_test = analysistest.target_under_test(env)
        actions = analysistest.target_actions(env)

        args = actions[2].content.split("\n")
        user_libs = []
        for arg in args:
            if arg.endswith(".o"):
                user_libs.append(arg)
        asserts.true(env, user_libs[-1].endswith("a_suffix.pic.o"), "liba_suffix.pic.o should be the last user library linked")

    return analysistest.end(env)

linking_suffix_test = analysistest.make(
    _linking_suffix_test_impl,
    attrs = {
        "is_linux": attr.bool(),
    },
)

def _additional_inputs_test_impl(ctx):
    env = analysistest.begin(ctx)

    if ctx.attr.is_linux:
        target_under_test = analysistest.target_under_test(env)
        actions = analysistest.target_actions(env)

        found = False
        for arg in actions[4].argv:
            if arg.find("-Wl,--script=") != -1:
                asserts.equals(env, "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/additional_script.txt", arg[13:])
                found = True
                break
        asserts.true(env, found, "Should have seen option --script=")

    return analysistest.end(env)

additional_inputs_test = analysistest.make(
    _additional_inputs_test_impl,
    attrs = {
        "is_linux": attr.bool(),
    },
)

def _build_failure_test_impl(ctx):
    env = analysistest.begin(ctx)

    if ctx.attr.message:
        asserts.expect_failure(env, ctx.attr.message)

    if ctx.attr.messages:
        for message in ctx.attr.messages:
            asserts.expect_failure(env, message)

    return analysistest.end(env)

build_failure_test = analysistest.make(
    _build_failure_test_impl,
    expect_failure = True,
    attrs = {
        "message": attr.string(),
        "messages": attr.string_list(),
    },
)

def _paths_test_impl(ctx):
    env = unittest.begin(ctx)

    asserts.false(env, _check_if_target_under_path(Label("//foo"), Label("//bar")))
    asserts.false(env, _check_if_target_under_path(Label("@foo//foo"), Label("@bar//bar")))
    asserts.false(env, _check_if_target_under_path(Label("//bar"), Label("@foo//bar")))
    asserts.true(env, _check_if_target_under_path(Label("@foo//bar"), Label("@foo//bar")))
    asserts.true(env, _check_if_target_under_path(Label("@foo//bar:bar"), Label("@foo//bar")))
    asserts.true(env, _check_if_target_under_path(Label("//bar:bar"), Label("//bar")))

    asserts.false(env, _check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar")))
    asserts.false(env, _check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar:__pkg__")))
    asserts.true(env, _check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar:__subpackages__")))
    asserts.true(env, _check_if_target_under_path(Label("@foo//bar:qux"), Label("@foo//bar:__pkg__")))

    asserts.false(env, _check_if_target_under_path(Label("@foo//bar"), Label("@foo//bar/baz:__subpackages__")))
    asserts.false(env, _check_if_target_under_path(Label("//bar"), Label("//bar/baz:__pkg__")))

    asserts.true(env, _check_if_target_under_path(Label("//foo/bar:baz"), Label("//:__subpackages__")))

    return unittest.end(env)

paths_test = unittest.make(_paths_test_impl)

def _debug_files_test_impl(ctx):
    env = analysistest.begin(ctx)

    target_under_test = analysistest.target_under_test(env)
    actual_files = []
    for debug_file in target_under_test[OutputGroupInfo].rule_impl_debug_files.to_list():
        actual_files.append(debug_file.basename)
    expected_files = ["bar_so_exports.txt", "bar_so_link_once_static_libs.txt", "foo_so_exports.txt", "foo_so_link_once_static_libs.txt", "binary_link_once_static_libs.txt"]
    asserts.equals(env, expected_files, actual_files)

    return analysistest.end(env)

debug_files_test = analysistest.make(_debug_files_test_impl)

def _runfiles_test_impl(ctx):
    env = analysistest.begin(ctx)
    if not ctx.attr.is_linux:
        return analysistest.end(env)

    target_under_test = analysistest.target_under_test(env)
    actual_files = []
    for runfile in target_under_test[DefaultInfo].default_runfiles.files.to_list():
        actual_files.append(runfile.basename)
    expected = [
        "renamed_so_file_copy.so",
        "libdirect_so_file.so",
    ]
    for expected_file in expected:
        asserts.true(env, expected_file in actual_files, expected_file + " not found in actual files:\n" + "\n".join(actual_files))

    return analysistest.end(env)

runfiles_test = analysistest.make(
    _runfiles_test_impl,
    attrs = {
        "is_linux": attr.bool(),
    },
)
