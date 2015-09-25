# Copyright 2015 The Bazel Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Build definitions for Closure stylesheets. Two files are produced: a minified
and obfuscated CSS file and a JS file defining a map between CSS classes and the
obfuscated class name. The stylesheet targets may be used in closure_js_binary
rules.

Both CSS and GSS files may be used with this rule.

Example:

  closure_stylesheet_library(
      name = "hello_css",
      srcs = ["hello.gss"].
  )

This rule will produce hello_css_combined.css and hello_css_renaming.js.
"""

_GSS_FILE_TYPE = FileType([".css", ".gss"])

def _impl(ctx):
  srcs = set(order="compile")
  for dep in ctx.attr.deps:
    srcs += dep.transitive_gss_srcs

  srcs += _GSS_FILE_TYPE.filter(ctx.files.srcs)

  args = [
      "--output-file",
      ctx.outputs.out.path,
      "--output-renaming-map",
      ctx.outputs.out_renaming.path,
      "--output-renaming-map-format",
      "CLOSURE_COMPILED",
      "--rename",
      "CLOSURE"
  ] + [src.path for src in srcs]

  ctx.action(
      inputs=list(srcs),
      outputs=[ctx.outputs.out, ctx.outputs.out_renaming],
      arguments=args,
      executable=ctx.executable._closure_stylesheets)

  return struct(
      files=set([ctx.outputs.out]),
      transitive_gss_srcs=srcs,
      transitive_js_externs=set(),
      transitive_js_srcs=[ctx.outputs.out_renaming])

# There are two outputs:
# - %{name}_combined.css: A minified and obfuscated CSS file.
# - %{name}_renaming.js:  A map from the original CSS class name to the
#                         obfuscated name. This file is used by
#                         closure_js_binary rules.
closure_stylesheet_library = rule(
    implementation=_impl,
    attrs={
        "srcs": attr.label_list(allow_files=_GSS_FILE_TYPE),
        "deps": attr.label_list(
            providers=["transitive_gss_srcs", "transitive_js_srcs"]),
        "_closure_stylesheets": attr.label(
            default=Label("//tools/build_rules/closure:closure_stylesheets"),
            executable=True),
    },
    outputs={
        "out": "%{name}_combined.css",
        "out_renaming": "%{name}_renaming.js"
    })
