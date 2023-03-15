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

load(":common/java/java_semantics.bzl", "semantics")

ProguardSpecProvider = _builtins.toplevel.ProguardSpecProvider

def _filter_provider(provider, *attrs):
    return [dep[provider] for attr in attrs for dep in attr if provider in dep]

def _validate_spec(ctx, spec_file):
    validated_proguard_spec = ctx.actions.declare_file(
        "validated_proguard/%s/%s_valid" % (ctx.label.name, spec_file.path),
    )

    toolchain = semantics.find_java_toolchain(ctx)

    args = ctx.actions.args()
    args.add("--path", spec_file)
    args.add("--output", validated_proguard_spec)

    ctx.actions.run(
        mnemonic = "ValidateProguard",
        progress_message = "Validating proguard configuration %{input}",
        executable = toolchain.proguard_allowlister,
        arguments = [args],
        inputs = [spec_file],
        outputs = [validated_proguard_spec],
        toolchain = Label(semantics.JAVA_TOOLCHAIN_TYPE),
    )

    return validated_proguard_spec

def validate_proguard_specs(ctx, proguard_specs = [], transitive_attrs = []):
    """
    Creates actions that validate Proguard specification and returns ProguardSpecProvider.

    Use transtive_attrs parameter to collect Proguard validations from `deps`,
    `runtime_deps`, `exports`, `plugins`, and `exported_plugins` attributes.

    Args:
      ctx: (RuleContext) Used to register the actions.
      proguard_specs: (list[File]) List of Proguard specs files.
      transitive_attrs: (list[list[Target]])  Attributes to collect transitive
        proguard validations from.
    Returns:
      (ProguardSpecProvider) A ProguardSpecProvider.
    """
    proguard_validations = _filter_provider(ProguardSpecProvider, *transitive_attrs)
    return ProguardSpecProvider(
        depset(
            [_validate_spec(ctx, spec_file) for spec_file in proguard_specs],
            transitive = [validation.specs for validation in proguard_validations],
        ),
    )
