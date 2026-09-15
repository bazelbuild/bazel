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
    for output, version, strict in [
        (ctx.outputs.out2, "2", "1"),
        (ctx.outputs.out3, "3", "1"),
        (ctx.outputs.out2_nonstrict, "2", "0"),
        (ctx.outputs.out3_nonstrict, "3", "0"),
    ]:
        if output:
            ctx.actions.expand_template(
                template = ctx.file.template,
                output = output,
                substitutions = {
                    "%VERSION%": version,
                    "%STRICT%": strict,
                },
                is_executable = True,
            )

expand_pyversion_template = rule(
    implementation = _expand_pyversion_template_impl,
    attrs = {
        "template": attr.label(
            allow_single_file = True,
            doc = "The input template file.",
        ),
        "out2": attr.output(doc = "The Python 2 strict wrapper."),
        "out3": attr.output(doc = "The Python 3 strict wrapper."),
        "out2_nonstrict": attr.output(
            doc = "The Python 2 non-strict wrapper.",
        ),
        "out3_nonstrict": attr.output(
            doc = "The Python 3 non-strict wrapper.",
        ),
    },
    doc = """\
Given the pywrapper template file, generates expansions for both versions of
Python and both levels of strictness.""",
)
