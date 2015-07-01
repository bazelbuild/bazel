# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Build definitions for JavaScript libraries. The library targets may be used
in closure_js_binary rules.

Example:

  closure_js_library(
      name = "hello_lib",
      srcs = ["hello.js"],
      deps = ["//third_party/javascript/closure_library"],
  )
"""

_JS_FILE_TYPE = FileType([".js"])

def _impl(ctx):
  externs = set(order="compile")
  srcs = set(order="compile")
  for dep in ctx.attr.deps:
    externs += dep.transitive_js_externs
    srcs += dep.transitive_js_srcs

  externs += _JS_FILE_TYPE.filter(ctx.files.externs)
  srcs += _JS_FILE_TYPE.filter(ctx.files.srcs)

  return struct(
      files=set(), transitive_js_externs=externs, transitive_js_srcs=srcs)

closure_js_library = rule(
    implementation=_impl,
    attrs={
        "externs": attr.label_list(allow_files=_JS_FILE_TYPE),
        "srcs": attr.label_list(allow_files=_JS_FILE_TYPE),
        "deps": attr.label_list(
            providers=["transitive_js_externs", "transitive_js_srcs"])
    })
