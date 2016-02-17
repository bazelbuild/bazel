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

"""Build definitions for JavaScript binaries compiled with the Closure Compiler.

A single file is produced with the _compiled.js suffix.

By default, the name of the entry point is assumed to be the same as that of the
build target. This behaviour may be overridden with the "main" attribute.

The optimization level may be set with the "compilation_level" attribute.
Supported values are: unobfuscated, simple, and advanced.

Example:

  closure_js_binary(
      name = "hello",
      compilation_level = "simple",
      language_in = "ecmascript6",
      language_out = "ecmascript3",
      externs = ["//third_party/javascript/google_cast/cast.js"],
      deps = [
          "@closure_library//:closure_library",
          ":hello_lib",
      ],
  )

This rule will produce hello_combined.js.
"""

_COMPILATION_LEVELS = {
  "whitespace_only": [
      "--compilation_level=WHITESPACE_ONLY",
      "--formatting=PRETTY_PRINT"
  ],
  "simple": ["--compilation_level=SIMPLE"],
  "advanced": ["--compilation_level=ADVANCED"]
}

_SUPPORTED_LANGUAGES = {
  "es3": ["ES3"],
  "ecmascript3": ["ECMASCRIPT3"],
  "es5": ["ES5"],
  "ecmascript5": ["ECMASCRIPT5"],
  "es5_strict": ["ES5_STRICT"],
  "ecmascript5_strict": ["ECMASCRIPT5_STRICT"],
  "es6": ["ES6"],
  "ecmascript6": ["ECMASCRIPT6"],
  "es6_strict": ["ES6_STRICT"],
  "ecmascript6_strict": ["ECMASCRIPT6_STRICT"],
  "es6_typed": ["ES6_TYPED"],
  "ecmascript6_typed": ["ECMASCRIPT6_TYPED"],
}

def _impl(ctx):
  externs = set(order="compile")
  srcs = set(order="compile")
  for dep in ctx.attr.deps:
    externs += dep.transitive_js_externs
    srcs += dep.transitive_js_srcs

  args = [
      "--closure_entry_point=%s" % ctx.attr.main,
      "--js_output_file=%s" % ctx.outputs.out.path,
      "--language_in=ECMASCRIPT5_STRICT",
      "--dependency_mode=LOOSE",
      "--warning_level=VERBOSE",
  ] + (["--js=%s" % src.path for src in srcs] +
       ["--externs=%s" % extern.path for extern in externs])

  # Set the compilation level.
  if ctx.attr.compilation_level in _COMPILATION_LEVELS:
    args += _COMPILATION_LEVELS[ctx.attr.compilation_level]
  else:
    fail("Invalid compilation_level '%s', expected one of %s" %
         (ctx.attr.compilation_level, _COMPILATION_LEVELS.keys()))

  # Set the language in.
  if ctx.attr.language_in in _SUPPORTED_LANGUAGES:
    args += "--language_in=" + _SUPPORTED_LANGUAGES[ctx.attr.language_in]
  else:
    fail("Invalid language_in '%s', expected one of %s" %
         (ctx.attr.language_in, _SUPPORTED_LANGUAGES.keys()))

  # Set the language out.
  if ctx.attr.language_out in _SUPPORTED_LANGUAGES:
    args += "--language_out=" + _SUPPORTED_LANGUAGES[ctx.attr.language_out]
  else:
    fail("Invalid language_out '%s', expected one of %s" %
         (ctx.attr.language_out, _SUPPORTED_LANGUAGES.keys()))

  ctx.action(
      inputs=list(srcs) + list(externs),
      outputs=[ctx.outputs.out],
      arguments=args,
      executable=ctx.executable._closure_compiler)

  return struct(files=set([ctx.outputs.out]))

closure_js_binary = rule(
    implementation=_impl,
    attrs={
        "deps": attr.label_list(
            allow_files=False,
            providers=["transitive_js_externs", "transitive_js_srcs"]),
        "main": attr.string(default="%{name}"),
        "compilation_level": attr.string(default="advanced"),
        "language_in": attr.string(default="ecmascript6"),
        "language_out": attr.string(default="ecmascript3"),
        "_closure_compiler": attr.label(
            default=Label("//external:closure_compiler_"),
            executable=True),
    },
    outputs={"out": "%{name}_combined.js"})
