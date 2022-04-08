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

"""cc_test Starlark implementation."""

load(":common/cc/cc_binary.bzl", "cc_binary_impl")
load(":common/paths.bzl", "paths")

# TODO(b/198254254): We need to do a wrapper around cc_test like for
# cc_binary, but for now it should work.
load(":common/cc/cc_binary_attrs.bzl", "cc_binary_attrs_with_aspects")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", "semantics")

cc_internal = _builtins.internal.cc_internal
platform_common = _builtins.toplevel.platform_common
testing = _builtins.toplevel.testing

_cc_test_attrs = dict(cc_binary_attrs_with_aspects)

# Update cc_test defaults:
_cc_test_attrs.update(
    _is_test = attr.bool(default = True),
    _apple_constraints = attr.label_list(
        default = [
            "@" + paths.join(semantics.get_platforms_root(), "os:ios"),
            "@" + paths.join(semantics.get_platforms_root(), "os:macos"),
            "@" + paths.join(semantics.get_platforms_root(), "os:tvos"),
            "@" + paths.join(semantics.get_platforms_root(), "os:watchos"),
        ],
    ),
    stamp = attr.int(values = [-1, 0, 1], default = 0),
    linkstatic = attr.bool(default = False),
    malloc = attr.label(
        default = Label("@//tools/cpp:cc_test_malloc"),
        allow_rules = ["cc_library"],
        # TODO(b/198254254): Add aspects. in progress
        aspects = [],
    ),
)

def _cc_test_impl(ctx):
    binary_info, cc_info, providers = cc_binary_impl(ctx, [])
    providers.append(testing.TestEnvironment(ctx.attr.env))
    providers.append(DefaultInfo(
        files = binary_info.files,
        runfiles = binary_info.runfiles,
        executable = binary_info.executable,
    ))
    return _handle_legacy_return(ctx, cc_info, providers)

def _handle_legacy_return(ctx, cc_info, providers):
    if cc_helper.has_target_constraints(ctx, ctx.attr._apple_constraints):
        # When built for Apple platforms, require the execution to be on a Mac.
        providers.append(testing.ExecutionInfo({"requires-darwin": ""}))
    if ctx.fragments.cpp.enable_legacy_cc_provider():
        # buildifier: disable=rule-impl-return
        return struct(
            cc = cc_internal.create_cc_provider(cc_info = cc_info),
            providers = providers,
        )
    else:
        return providers

def _impl(ctx):
    cpp_config = ctx.fragments.cpp
    cc_test_info = ctx.toolchains["@//tools/cpp:test_runner_toolchain_type"].cc_test_info

    if not cpp_config.experimental_platform_cc_test() or cc_test_info.use_legacy_cc_test:
        # This is the "legacy" cc_test flow
        return _cc_test_impl(ctx)

    binary_info, cc_info, providers = cc_binary_impl(ctx, cc_test_info.linkopts)

    test_providers = cc_test_info.get_runner.func(
        ctx,
        binary_info,
        **cc_test_info.get_runner.args
    )
    providers.extend(test_providers)
    return _handle_legacy_return(ctx, cc_info, providers)

def make_cc_test(with_linkstatic = False):
    _cc_test_attrs.update(
        _linkstatic_explicitly_set = attr.bool(default = with_linkstatic),
    )
    return rule(
        name = "cc_test",
        implementation = _impl,
        attrs = _cc_test_attrs,
        outputs = {
            # TODO(b/198254254): Handle case for windows.
            "stripped_binary": "%{name}.stripped",
            "dwp_file": "%{name}.dwp",
        },
        fragments = ["google_cpp", "cpp"],
        exec_groups = {
            "cpp_link": exec_group(copy_from_rule = True),
        },
        toolchains = [
            "@//tools/cpp:toolchain_type",
            "@//tools/cpp:test_runner_toolchain_type",
        ],
        incompatible_use_toolchain_transition = True,
        test = True,
    )

cc_test_explicit_linkstatic = make_cc_test(with_linkstatic = True)
cc_test_default_linkstatic = make_cc_test(with_linkstatic = False)

def cc_test_wrapper(**kwargs):
    if "linkstatic" in kwargs:
        cc_test_explicit_linkstatic(**kwargs)
    else:
        cc_test_default_linkstatic(**kwargs)
