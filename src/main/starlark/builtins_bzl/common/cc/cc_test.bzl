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

load(":common/cc/cc_binary.bzl", "cc_binary_attrs", "cc_binary_impl")

testing = _builtins.toplevel.testing

_cc_test_attrs = dict(cc_binary_attrs)

# Update other cc_test defaults:
_cc_test_attrs.update(
    _is_test = attr.bool(default = True),
    stamp = attr.int(default = 0),
    linkstatic = attr.bool(default = False),
    malloc = attr.label(
        default = Label("@//tools/cpp:cc_test_malloc"),
        allow_rules = ["cc_library"],
        # TODO(b/198254254): Add aspects. in progress
        aspects = [],
    ),
)

def _cc_test_impl(ctx):
    binary_info = cc_binary_impl(ctx)
    env = testing.TestEnvironment(ctx.attr.env)
    binary_info.append(env)
    return binary_info

cc_test = rule(
    implementation = _cc_test_impl,
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
    ],
    incompatible_use_toolchain_transition = True,
    test = True,
)
