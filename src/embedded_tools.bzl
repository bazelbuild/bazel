# pylint: disable=g-bad-file-header
# Copyright 2017 The Bazel Authors. All rights reserved.
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
"""Contains Starlark rules used to build the embedded_tools.zip."""

def _srcsfile(ctx):
    ctx.actions.write(
        output = ctx.outputs.out,
        content = "\n".join([f.path for f in ctx.files.srcs]),
    )

srcsfile = rule(
    implementation = _srcsfile,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(mandatory = True),
    },
)
