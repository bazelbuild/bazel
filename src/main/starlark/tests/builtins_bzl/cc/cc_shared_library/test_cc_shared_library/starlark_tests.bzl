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

        target_action = None
        for action in actions:
            if action.mnemonic == "FileWrite":
                target_action = action
                break
        args = target_action.content.split("\n")
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
        target_action = None
        for action in actions:
            if action.mnemonic == "CppLink":
                target_action = action
                break
        for arg in target_action.argv:
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

    expected_files = [
        "bar_so_exports.txt",
        "bar_so_link_once_static_libs.txt",
        "diff_pkg_so_exports.txt",
        "diff_pkg_so_link_once_static_libs.txt",
        "foo_so_exports.txt",
        "foo_so_link_once_static_libs.txt",
        "binary_link_once_static_libs.txt",
    ]
    asserts.equals(env, expected_files, actual_files)

    return analysistest.end(env)

debug_files_test = analysistest.make(_debug_files_test_impl)

def _runfiles_test_impl(ctx):
    env = analysistest.begin(ctx)
    if not ctx.attr.is_linux:
        return analysistest.end(env)

    target_under_test = analysistest.target_under_test(env)
    expected_suffixes = [
        "libfoo_so.so",
        "libbar_so.so",
        "libdiff_pkg_so.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary_Slibfoo_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary_Slibbar_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary3_Slibdiff_Upkg_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary/renamed_so_file_copy.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary/libdirect_so_file.so",
    ]
    for runfile in target_under_test[DefaultInfo].default_runfiles.files.to_list():
        # Ignore Python runfiles
        if "python" in runfile.path:
            continue
        found_suffix = False
        for expected_suffix in expected_suffixes:
            if runfile.path.endswith(expected_suffix):
                found_suffix = True
                break
        asserts.true(env, found_suffix, runfile.path + " not found in expected suffixes:\n" + "\n".join(expected_suffixes))

    return analysistest.end(env)

runfiles_test = analysistest.make(
    _runfiles_test_impl,
    attrs = {
        "is_linux": attr.bool(),
    },
)

def _interface_library_output_group_test_impl(ctx):
    env = analysistest.begin(ctx)
    if not ctx.attr.is_windows:
        return analysistest.end(env)

    target_under_test = analysistest.target_under_test(env)
    actual_files = []
    for interface_library in target_under_test[OutputGroupInfo].interface_library.to_list():
        actual_files.append(interface_library.basename)
    expected_files = ["foo_so.if.lib"]
    asserts.equals(env, expected_files, actual_files)

    return analysistest.end(env)

interface_library_output_group_test = analysistest.make(
    _interface_library_output_group_test_impl,
    attrs = {
        "is_windows": attr.bool(),
    },
)

def _check_linking_action_lib_parameters_test_impl(ctx):
    env = analysistest.begin(ctx)

    actions = analysistest.target_actions(env)

    target_action = None
    for action in actions:
        if action.mnemonic == "FileWrite":
            target_action = action
            break
    args = target_action.content.split("\n")
    for arg in args:
        for bad_lib_entry in ctx.attr.libs_that_shouldnt_be_present:
            asserts.true(env, arg.find("{}.".format(bad_lib_entry)) == -1, "Should not have seen library `{}` in command line".format(arg))

    return analysistest.end(env)

check_linking_action_lib_parameters_test = analysistest.make(
    _check_linking_action_lib_parameters_test_impl,
    attrs = {
        "libs_that_shouldnt_be_present": attr.string_list(),
    },
)

AspectCcInfo = provider("Takes a cc_info.", fields = {"cc_info": "cc_info"})
WrappedCcInfo = provider("Takes a cc_info.", fields = {"cc_info": "cc_info"})

def _forwarding_cc_lib_aspect_impl(target, ctx):
    cc_info = target[WrappedCcInfo].cc_info
    linker_inputs = []
    owner = ctx.label.relative(ctx.label.name + ".custom")
    for linker_input in cc_info.linking_context.linker_inputs.to_list():
        if linker_input.owner == ctx.label.relative("indirect_dep"):
            linker_inputs.append(cc_common.create_linker_input(
                owner = owner,
                libraries = depset(linker_input.libraries),
            ))
        else:
            linker_inputs.append(linker_input)
    cc_info = CcInfo(
        compilation_context = cc_info.compilation_context,
        linking_context = cc_common.create_linking_context(
            linker_inputs = depset(linker_inputs),
        ),
    )
    return [
        AspectCcInfo(cc_info = cc_info),
        CcSharedLibraryHintInfo(
            owners = [owner],
        ),
    ]

forwarding_cc_lib_aspect = aspect(
    implementation = _forwarding_cc_lib_aspect_impl,
    required_providers = [WrappedCcInfo],
    provides = [AspectCcInfo, CcSharedLibraryHintInfo],
)

def _wrapped_cc_lib_impl(ctx):
    return [WrappedCcInfo(cc_info = ctx.attr.deps[0][CcInfo]), ProtoInfo()]

wrapped_cc_lib = rule(
    implementation = _wrapped_cc_lib_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
    },
    provides = [WrappedCcInfo, ProtoInfo],
)

def _forwarding_cc_lib_impl(ctx):
    hints = CcSharedLibraryHintInfo(attributes = ["deps"])
    if ctx.attr.deps:
        return [ctx.attr.deps[0][AspectCcInfo].cc_info, hints]
    else:
        return [ctx.attr.do_not_follow_deps[0][AspectCcInfo].cc_info, hints]

forwarding_cc_lib = rule(
    implementation = _forwarding_cc_lib_impl,
    attrs = {
        "deps": attr.label_list(providers = [WrappedCcInfo], aspects = [forwarding_cc_lib_aspect]),
        "do_not_follow_deps": attr.label_list(providers = [WrappedCcInfo], aspects = [forwarding_cc_lib_aspect]),
    },
    provides = [CcInfo, CcSharedLibraryHintInfo],
)

def _nocode_cc_lib_impl(ctx):
    linker_input = cc_common.create_linker_input(
        owner = ctx.label,
        additional_inputs = depset([ctx.files.additional_inputs[0]]),
    )
    cc_info = CcInfo(linking_context = cc_common.create_linking_context(linker_inputs = depset([linker_input])))
    return [cc_common.merge_cc_infos(cc_infos = [cc_info, ctx.attr.deps[0][CcInfo]])]

nocode_cc_lib = rule(
    implementation = _nocode_cc_lib_impl,
    attrs = {
        "additional_inputs": attr.label_list(allow_files = True),
        "deps": attr.label_list(),
    },
    provides = [CcInfo],
)
