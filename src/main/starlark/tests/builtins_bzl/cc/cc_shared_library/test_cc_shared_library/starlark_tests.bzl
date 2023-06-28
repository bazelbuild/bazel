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

load("@bazel_skylib//lib:paths.bzl", "paths")
load("@rules_testing//lib:truth.bzl", "matching")
load("@rules_testing//lib:analysis_test.bzl", "analysis_test")

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

def _linking_order_test_impl(env, target):
    if env.ctx.target_platform_has_constraint(env.ctx.attr._is_linux[platform_common.ConstraintValueInfo]):
        target_action = None
        for action in target.actions:
            if action.mnemonic == "FileWrite":
                target_action = action
                break

        args = target_action.content.split("\n")
        user_libs = [paths.basename(arg) for arg in args if arg.endswith(".o")]

        env.expect.that_collection(user_libs).contains_at_least_predicates([
            matching.contains("foo.pic.o"),
            matching.contains("baz.pic.o"),
        ]).in_order()

        env.expect.that_collection(args).contains_at_least([
            "-lprivate_lib_so",
        ])

        env.expect.where(
            detail = "liba_suffix.pic.o should be the last user library linked",
        ).that_str(user_libs[-1]).equals("a_suffix.pic.o")

        # qux2 is a LINKABLE_MORE_THAN_ONCE library which is enabled by semantics.
        # It might not be present but if it is we want to test it's in the right
        # place in the linking command line before libbar
        if "qux2.pic.o" in user_libs:
            found_bar = False
            for arg in args:
                if "-lbar_so" in arg:
                    found_bar = True
                elif "qux2.pic.o" in arg:
                    env.expect.where(
                        detail = "qux2 should come before bar in command line",
                    ).that_bool(found_bar).equals(False)
            env.expect.where(
                detail = "should have seen bar in command line",
            ).that_bool(found_bar).equals(True)

def _linking_order_test_macro(name, target):
    analysis_test(
        name = name,
        impl = _linking_order_test_impl,
        target = target,
        attrs = {
            "_is_linux": attr.label(default = "@platforms//os:linux"),
        },
    )

linking_order_test = _linking_order_test_macro

def _additional_inputs_test_impl(env, target):
    if env.ctx.target_platform_has_constraint(env.ctx.attr._is_linux[platform_common.ConstraintValueInfo]):
        found = False
        target_action = None
        for action in target.actions:
            if action.mnemonic == "CppLink":
                target_action = action
                break
        for arg in target_action.argv:
            if arg.find("-Wl,--script=") != -1:
                env.expect.that_str(
                    "src/main/starlark/tests/builtins_bzl/cc/cc_shared_library/test_cc_shared_library/additional_script.txt",
                ).equals(arg[13:])
                found = True
                break
        env.expect.where(
            detail = "Should have seen option --script=",
        ).that_bool(found).equals(True)

def _additional_inputs_test_macro(name, target):
    analysis_test(
        name = name,
        impl = _additional_inputs_test_impl,
        target = target,
        attrs = {
            "_is_linux": attr.label(default = "@platforms//os:linux"),
        },
    )

additional_inputs_test = _additional_inputs_test_macro

def _build_failure_test_impl(env, target):
    if env.ctx.attr._message:
        env.expect.that_target(target).failures().contains_predicate(matching.contains(env.ctx.attr._message))

    if env.ctx.attr._messages:
        for message in env.ctx.attr._messages:
            env.expect.that_target(target).failures().contains_predicate(matching.contains(message))

def _build_failure_test_macro(name, target, message = "", messages = []):
    analysis_test(
        name = name,
        impl = _build_failure_test_impl,
        target = target,
        expect_failure = True,
        attrs = {
            "_message": attr.string(default = message),
            "_messages": attr.string_list(default = messages),
        },
    )

build_failure_test = _build_failure_test_macro

def _paths_test_impl(env, _):
    env.expect.that_bool(_check_if_target_under_path(Label("//foo"), Label("//bar"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//foo"), Label("@bar//bar"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("//bar"), Label("@foo//bar"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar"), Label("@foo//bar"))).equals(True)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar:bar"), Label("@foo//bar"))).equals(True)
    env.expect.that_bool(_check_if_target_under_path(Label("//bar:bar"), Label("//bar"))).equals(True)

    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar:__pkg__"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar/baz"), Label("@foo//bar:__subpackages__"))).equals(True)
    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar:qux"), Label("@foo//bar:__pkg__"))).equals(True)

    env.expect.that_bool(_check_if_target_under_path(Label("@foo//bar"), Label("@foo//bar/baz:__subpackages__"))).equals(False)
    env.expect.that_bool(_check_if_target_under_path(Label("//bar"), Label("//bar/baz:__pkg__"))).equals(False)

    env.expect.that_bool(_check_if_target_under_path(Label("//foo/bar:baz"), Label("//:__subpackages__"))).equals(True)

def _paths_test_macro(name):
    native.cc_library(
        name = "dummy",
    )
    analysis_test(
        name = name,
        impl = _paths_test_impl,
        target = ":dummy",
    )

paths_test = _paths_test_macro

def _runfiles_test_impl(env, target):
    if not env.ctx.target_platform_has_constraint(env.ctx.attr._is_linux[platform_common.ConstraintValueInfo]):
        return

    expected_basenames = [
        "libfoo_so.so",
        "libbar_so.so",
        "libdiff_pkg_so.so",
        "libprivate_lib_so.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary_Slibfoo_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary_Slibbar_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary3_Slibdiff_Upkg_Uso.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary/renamed_so_file_copy.so",
        "Smain_Sstarlark_Stests_Sbuiltins_Ubzl_Scc_Scc_Ushared_Ulibrary_Stest_Ucc_Ushared_Ulibrary/libdirect_so_file.so",
    ]
    for runfile in target[DefaultInfo].default_runfiles.files.to_list():
        # Ignore Python runfiles
        if "python" in runfile.path:
            continue

        found_basename = False
        for expected_basename in expected_basenames:
            if runfile.path.endswith(expected_basename):
                found_basename = True
                break

        env.expect.where(
            detail = runfile.path + " not found in expected basenames:\n" + "\n".join(expected_basenames),
        ).that_bool(found_basename).equals(True)

def _runfiles_test_macro(name, target):
    analysis_test(
        name = name,
        impl = _runfiles_test_impl,
        target = target,
        attrs = {
            "_is_linux": attr.label(default = "@platforms//os:linux"),
        },
    )

runfiles_test = _runfiles_test_macro

def _interface_library_output_group_test_impl(env, target):
    if not env.ctx.target_platform_has_constraint(env.ctx.attr._is_windows[platform_common.ConstraintValueInfo]):
        return

    actual_files = [interface_library.basename for interface_library in target[OutputGroupInfo].interface_library.to_list()]
    env.expect.that_collection(actual_files).contains_exactly_predicates([
        matching.contains("foo_so.if.lib"),
    ])

def _interface_library_output_group_test_macro(name, target):
    analysis_test(
        name = name,
        impl = _interface_library_output_group_test_impl,
        target = target,
        attrs = {
            "_is_windows": attr.label(default = "@platforms//os:windows"),
        },
    )

interface_library_output_group_test = _interface_library_output_group_test_macro

def _check_linking_action_lib_parameters_test_impl(env, target):
    target_action = None
    for action in target.actions:
        if action.mnemonic == "FileWrite":
            target_action = action
            break

    args = target_action.content.split("\n")
    for arg in args:
        for bad_lib_entry in env.ctx.attr._libs_that_shouldnt_be_present:
            env.expect.where(
                detail = "Should not have seen library `{}` in command line".format(arg),
            ).that_int(arg.find("{}.".format(bad_lib_entry))).equals(-1)

def _check_linking_action_lib_parameters_test_macro(name, target, libs_that_shouldnt_be_present):
    analysis_test(
        name = name,
        impl = _check_linking_action_lib_parameters_test_impl,
        target = target,
        attrs = {
            "_libs_that_shouldnt_be_present": attr.string_list(default = libs_that_shouldnt_be_present),
        },
    )

check_linking_action_lib_parameters_test = _check_linking_action_lib_parameters_test_macro

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
    providers = [AspectCcInfo(cc_info = cc_info)]
    if hasattr(cc_common, "CcSharedLibraryHintInfo"):
        providers.append(cc_common.CcSharedLibraryHintInfo(
            owners = [owner],
        ))

    return providers

forwarding_cc_lib_aspect = aspect(
    implementation = _forwarding_cc_lib_aspect_impl,
    required_providers = [WrappedCcInfo],
    provides = [AspectCcInfo, CcSharedLibraryHintInfo],
)

def _wrapped_cc_lib_impl(ctx):
    descriptor_set = ctx.actions.declare_file("fake.descriptor_set")
    ctx.actions.write(descriptor_set, "")
    return [WrappedCcInfo(cc_info = ctx.attr.deps[0][CcInfo]), ProtoInfo(srcs = [], deps = [], descriptor_set = descriptor_set)]

wrapped_cc_lib = rule(
    implementation = _wrapped_cc_lib_impl,
    attrs = {
        "deps": attr.label_list(providers = [CcInfo]),
    },
    provides = [WrappedCcInfo, ProtoInfo],
)

def _forwarding_cc_lib_impl(ctx):
    hints = cc_common.CcSharedLibraryHintInfo(attributes = ["deps"])
    if ctx.attr.deps:
        return [ctx.attr.deps[0][AspectCcInfo].cc_info, hints]
    else:
        return [ctx.attr.do_not_follow_deps[0][AspectCcInfo].cc_info, hints]

def _get_provides_list():
    provides = [CcInfo]
    if hasattr(cc_common, "CcSharedLibraryHintInfo"):
        provides.append(cc_common.CcSharedLibraryHintInfo)
    return provides

forwarding_cc_lib = rule(
    implementation = _forwarding_cc_lib_impl,
    attrs = {
        "deps": attr.label_list(providers = [WrappedCcInfo], aspects = [forwarding_cc_lib_aspect]),
        "do_not_follow_deps": attr.label_list(providers = [WrappedCcInfo], aspects = [forwarding_cc_lib_aspect]),
    },
    provides = _get_provides_list(),
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

def _exports_test_impl(env, target):
    actual = list(target[CcSharedLibraryInfo].exports)

    # Remove the @ prefix on Bazel
    for i in range(len(actual)):
        if actual[i][0] == "@":
            actual[i] = actual[i][1:]
    expected = env.ctx.attr._targets_that_should_be_claimed_to_be_exported
    env.expect.where(
        detail = "Exports lists do not match.",
    ).that_collection(actual).contains_exactly(expected).in_order()

def _exports_test_macro(name, target, targets_that_should_be_claimed_to_be_exported):
    analysis_test(
        name = name,
        impl = _exports_test_impl,
        target = target,
        attrs = {
            "_targets_that_should_be_claimed_to_be_exported": attr.string_list(default = targets_that_should_be_claimed_to_be_exported),
        },
    )

exports_test = _exports_test_macro
