# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Implementation for the java_package_configuration rule"""

load(":common/java/java_helper.bzl", "helper")

PackageSpecificationInfo = _builtins.toplevel.PackageSpecificationInfo

JavaPackageConfigurationInfo = provider(
    "A provider for Java per-package configuration",
    fields = [
        "data",
        "javac_opts",
        "javac_opts_list",
        "matches",
        "package_specs",
    ],
)

def _matches(package_specs, label):
    for spec in package_specs:
        if spec.contains(label):
            return True
    return False

def _rule_impl(ctx):
    javacopts = [
        token
        for opt in ctx.attr.javacopts
        for token in ctx.tokenize(ctx.expand_location(opt, targets = ctx.attr.data))
    ]
    javacopts_depset = helper.detokenize_javacopts(javacopts)
    package_specs = [package[PackageSpecificationInfo] for package in ctx.attr.packages]
    return [
        DefaultInfo(),
        JavaPackageConfigurationInfo(
            data = depset(ctx.files.data),
            javac_opts = lambda as_depset: javacopts_depset if as_depset else javacopts,
            javac_opts_list = javacopts,
            matches = lambda label: _matches(package_specs, label),
            package_specs = package_specs,
        ),
    ]

java_package_configuration = rule(
    implementation = _rule_impl,
    attrs = {
        "packages": attr.label_list(
            cfg = "exec",
            providers = [PackageSpecificationInfo],
        ),
        "javacopts": attr.string_list(),
        "data": attr.label_list(allow_files = True),
        "output_licenses": attr.license() if hasattr(attr, "license") else attr.string_list(),
    },
)
