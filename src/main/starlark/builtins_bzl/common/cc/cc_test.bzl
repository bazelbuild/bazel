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
load(":common/cc/cc_binary_attrs.bzl", "cc_binary_attrs_with_aspects", "cc_binary_attrs_without_aspects")
load(":common/cc/cc_helper.bzl", "cc_helper")
load(":common/cc/semantics.bzl", "semantics")

cc_internal = _builtins.internal.cc_internal
platform_common = _builtins.toplevel.platform_common
testing = _builtins.toplevel.testing

def _cc_test_impl(ctx):
    binary_info, cc_info, providers = cc_binary_impl(ctx, [])
    test_env = {}
    test_env.update(cc_helper.get_expanded_env(ctx, {}))

    coverage_runfiles, coverage_env = semantics.get_coverage_env(ctx)

    runfiles_list = [binary_info.runfiles]
    if coverage_runfiles:
        runfiles_list.append(coverage_runfiles)

    runfiles = ctx.runfiles()
    runfiles = runfiles.merge_all(runfiles_list)

    test_env.update(coverage_env)
    providers.append(testing.TestEnvironment(
        environment = test_env,
        inherited_environment = ctx.attr.env_inherit,
    ))
    providers.append(DefaultInfo(
        files = binary_info.files,
        runfiles = runfiles,
        executable = binary_info.executable,
    ))

    if cc_helper.has_target_constraints(ctx, ctx.attr._apple_constraints):
        # When built for Apple platforms, require the execution to be on a Mac.
        providers.append(testing.ExecutionInfo({"requires-darwin": ""}))
    return _handle_legacy_return(ctx, cc_info, providers)

def _handle_legacy_return(ctx, cc_info, providers):
    if ctx.fragments.cpp.enable_legacy_cc_provider():
        # buildifier: disable=rule-impl-return
        return struct(
            cc = cc_internal.create_cc_provider(cc_info = cc_info),
            providers = providers,
        )
    else:
        return providers

def _impl(ctx):
    if semantics.should_use_legacy_cc_test(ctx):
        # This is the "legacy" cc_test flow
        return _cc_test_impl(ctx)

    cc_test_info = ctx.attr._test_toolchain.cc_test_info

    binary_info, cc_info, providers = cc_binary_impl(ctx, cc_test_info.linkopts)
    processed_environment = cc_helper.get_expanded_env(ctx, {})

    test_providers = cc_test_info.get_runner.func(
        ctx,
        binary_info,
        processed_environment = processed_environment,
        **cc_test_info.get_runner.args
    )
    providers.extend(test_providers)
    return _handle_legacy_return(ctx, cc_info, providers)

def _make_cc_test(with_linkstatic = False, with_aspects = False):
    _cc_test_attrs = None
    if with_aspects:
        _cc_test_attrs = dict(cc_binary_attrs_with_aspects)
    else:
        _cc_test_attrs = dict(cc_binary_attrs_without_aspects)

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
        _windows_constraints = attr.label_list(
            default = [
                "@" + paths.join(semantics.get_platforms_root(), "os:windows"),
            ],
        ),
        # Starlark tests don't get `env_inherit` by default.
        env_inherit = attr.string_list(),
        stamp = attr.int(values = [-1, 0, 1], default = 0),
        linkstatic = attr.bool(default = False),
    )
    _cc_test_attrs.update(semantics.get_test_malloc_attr())
    _cc_test_attrs.update(semantics.get_test_toolchain_attr())
    _cc_test_attrs.update(semantics.get_coverage_attrs())

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
        toolchains = cc_helper.use_cpp_toolchain(),
        incompatible_use_toolchain_transition = True,
        test = True,
    )

_cc_test_variants = struct(
    with_aspects = struct(
        explicit_linkstatic = _make_cc_test(with_linkstatic = True, with_aspects = True),
        default_linkstatic = _make_cc_test(with_aspects = True),
    ),
    without_aspects = struct(
        explicit_linkstatic = _make_cc_test(with_linkstatic = True),
        default_linkstatic = _make_cc_test(),
    ),
)

def cc_test_wrapper(**kwargs):
    """Entry point for cc_test rules.

    This avoids propagating aspects on certain attributes if dynamic_deps attribute is unset.

    It also serves to detect if the `linkstatic` attribute was explicitly set or not.
    This is to workaround a deficiency in Starlark attributes.
    (See: https://github.com/bazelbuild/bazel/issues/14434)

    Args:
        **kwargs: Arguments suitable for cc_test.
    """
    cc_test_aspects = None

    # Propagate an aspect if dynamic_deps attribute is specified.
    if "dynamic_deps" in kwargs and cc_helper.is_non_empty_list_or_select(kwargs["dynamic_deps"], "dynamic_deps"):
        cc_test_aspects = _cc_test_variants.with_aspects
    else:
        cc_test_aspects = _cc_test_variants.without_aspects

    if "linkstatic" in kwargs:
        cc_test_aspects.explicit_linkstatic(**kwargs)
    else:
        cc_test_aspects.default_linkstatic(**kwargs)
