# Copyright 2019 The Bazel Authors. All rights reserved.
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

"""Utilities for the @bazel_tools//tools/python package.

This file does not access any Python-rules-specific logic, and is therefore
less likely to be broken by Python-related changes. That in turn means this
file is less likely to cause bootstrapping issues.
"""

def _expand_pyversion_template_impl(ctx):
    if ctx.outputs.out2:
        ctx.actions.expand_template(
            template = ctx.file.template,
            output = ctx.outputs.out2,
            substitutions = {"%VERSION%": "2"},
            is_executable = True,
        )
    if ctx.outputs.out3:
        ctx.actions.expand_template(
            template = ctx.file.template,
            output = ctx.outputs.out3,
            substitutions = {"%VERSION%": "3"},
            is_executable = True,
        )

expand_pyversion_template = rule(
    implementation = _expand_pyversion_template_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            doc = "The input template file.",
        ),
        "out2": attr.output(doc = """\
The output file produced by substituting "%VERSION%" with "2"."""),
        "out3": attr.output(doc = """\
The output file produced by substituting "%VERSION%" with "3"."""),
    },
    doc = """\
Given a template file, generates two expansions by replacing the substring
"%VERSION%" with "2" and "3".""",
)
