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

"""Rules to make the default javacopts available as a Java API."""

def _default_javacopts(ctx):
   javacopts = java_common.default_javac_opts(
       ctx, java_toolchain_attr = "_java_toolchain")
   ctx.template_action(
      template = ctx.file.template,
      output = ctx.outputs.out,
      substitutions = {
          "%javacopts%": '"%s"' % '", "'.join(javacopts),
      }
   )

default_javacopts = rule(
    implementation=_default_javacopts,
    attrs={
        "template": attr.label(
            mandatory=True,
            allow_files=True,
            single_file=True,
        ),
        "out": attr.output(mandatory=True),
        "_java_toolchain": attr.label(
            default = Label("//tools/jdk:current_java_toolchain"),
        ),
    },
)
"""Makes the default javacopts available as a Java API.

Args:
  template: The template file to expand, replacing %javacopts% with a quoted
    comma-separated list of the default javac flags.
  out: The destination of the expanded file.

"""
