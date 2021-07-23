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

"""
Proguard
"""

load(":common/rule_util.bzl", "create_dep")

ProguardSpecProvider = _builtins.toplevel.ProguardSpecProvider

def _get_attr_safe(ctx, attr, default):
    return getattr(ctx.attr, attr) if hasattr(ctx.attr, attr) else default

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _collect_transitive_proguard_specs(ctx):
    attrs = [
        ctx.attr.deps,
        _get_attr_safe(ctx, "runtime_deps", []),
        _get_attr_safe(ctx, "exports", []),
        _get_attr_safe(ctx, "plugins", []),
        _get_attr_safe(ctx, "exported_plugins", []),
    ]

    return [proguard.specs for proguard in _filter_provider(ProguardSpecProvider, *attrs)]

def _validate_spec(ctx, spec_file):
    validated_proguard_spec = ctx.actions.declare_file(
        "validated_proguard/%s/%s_valid" % (ctx.label.name, spec_file.path),
    )

    args = ctx.actions.args()
    args.add("--path", spec_file)
    args.add("--output", validated_proguard_spec)

    ctx.actions.run(
        mnemonic = "ValidateProguard",
        progress_message = "Validating proguard configuration %{input}",
        executable = ctx.executable._proguard_allowlister,
        arguments = [args],
        inputs = [spec_file],
        outputs = [validated_proguard_spec],
    )

    return validated_proguard_spec

def _validate_proguard_specs_impl(ctx):
    if ctx.files.proguard_specs and not hasattr(ctx.attr, "_proguard_allowlister"):
        fail("_proguard_allowlister is required to use proguard_specs")

    return ProguardSpecProvider(
        depset(
            [_validate_spec(ctx, spec_file) for spec_file in ctx.files.proguard_specs],
            transitive = _collect_transitive_proguard_specs(ctx),
        ),
    )

VALIDATE_PROGUARD_SPECS = create_dep(
    _validate_proguard_specs_impl,
    {
        "proguard_specs": attr.label_list(allow_files = True),
    },
    [],
)
