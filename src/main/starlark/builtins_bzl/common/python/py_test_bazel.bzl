# Copyright 2022 The Bazel Authors. All rights reserved.
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
"""Rule implementation of py_test for Bazel."""

load(":common/python/attributes.bzl", "AGNOSTIC_TEST_ATTRS")
load(":common/python/common.bzl", "maybe_add_test_execution_info")
load(
    ":common/python/py_executable_bazel.bzl",
    "create_executable_rule",
    "py_executable_bazel_impl",
)
load(":common/python/semantics.bzl", "TOOLS_REPO")

_BAZEL_PY_TEST_ATTRS = {
    # This *might* be a magic attribute to help C++ coverage work. There's no
    # docs about this; see TestActionBuilder.java
    "_lcov_merger": attr.label(
        default = configuration_field(fragment = "coverage", name = "output_generator"),
        cfg = "exec",
        executable = True,
    ),
    # This *might* be a magic attribute to help C++ coverage work. There's no
    # docs about this; see TestActionBuilder.java
    "_collect_cc_coverage": attr.label(
        default = "@" + TOOLS_REPO + "//tools/test:collect_cc_coverage",
        executable = True,
        cfg = "exec",
    ),
}

def _py_test_impl(ctx):
    providers = py_executable_bazel_impl(
        ctx = ctx,
        is_test = True,
        inherited_environment = ctx.attr.env_inherit,
    )
    maybe_add_test_execution_info(providers, ctx)
    return providers

py_test = create_executable_rule(
    implementation = _py_test_impl,
    attrs = AGNOSTIC_TEST_ATTRS | _BAZEL_PY_TEST_ATTRS,
    test = True,
)
