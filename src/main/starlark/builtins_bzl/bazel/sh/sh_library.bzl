# Copyright 2024 The Bazel Authors. All rights reserved.
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

coverage_common = _builtins.toplevel.coverage_common

def _sh_library_impl(ctx):
    transitive_files = []
    for target in ctx.attr.srcs:
        transitive_files.append(target[DefaultInfo].files)
    for target in ctx.attr.deps:
        transitive_files.append(target[DefaultInfo].files)
    for target in ctx.attr.data:
        transitive_files.append(target[DefaultInfo].files)
    files = depset(transitive = transitive_files)

    runfiles = ctx.runfiles(transitive_files = files, collect_default = True)

    instrumented_files_info = coverage_common.instrumented_files_info(
        ctx,
        source_attributes = ["srcs"],
        dependency_attributes = ["deps", "data"],
    )

    return [
        DefaultInfo(
            files = files,
            runfiles = runfiles,
        ),
        instrumented_files_info,
    ]

sh_library = rule(
    _sh_library_impl,
    doc = """
<p>
  The main use for this rule is to aggregate together a logical
  "library" consisting of related scripts&mdash;programs in an
  interpreted language that does not require compilation or linking,
  such as the Bourne shell&mdash;and any data those programs need at
  run-time. Such "libraries" can then be used from
  the <code>data</code> attribute of one or
  more <code>sh_binary</code> rules.
</p>

<p>
  You can use the <a href="${link filegroup}"><code>filegroup</code></a> rule to aggregate data
  files.
</p>

<p>
  In interpreted programming languages, there's not always a clear
  distinction between "code" and "data": after all, the program is
  just "data" from the interpreter's point of view. For this reason
  this rule has three attributes which are all essentially equivalent:
  <code>srcs</code>, <code>deps</code> and <code>data</code>.
  The current implementation does not distinguish between the elements of these lists.
  All three attributes accept rules, source files and generated files.
  It is however good practice to use the attributes for their usual purpose (as with other rules).
</p>

<h4 id="sh_library_examples">Examples</h4>

<pre class="code">
sh_library(
    name = "foo",
    data = [
        ":foo_service_script",  # an sh_binary with srcs
        ":deploy_foo",  # another sh_binary with srcs
    ],
)
</pre>
""",
    attrs = {
        "srcs": attr.label_list(
            allow_files = True,
            doc = """
The list of input files.
<p>
  This attribute should be used to list shell script source files that belong to
  this library. Scripts can load other scripts using the shell's <code>source</code>
  or <code>.</code> command.
</p>
""",
        ),
        "data": attr.label_list(
            allow_files = True,
            flags = ["SKIP_CONSTRAINTS_OVERRIDE"],
        ),
        "deps": attr.label_list(
            allow_rules = ["sh_library"],
            doc = """
The list of "library" targets to be aggregated into this target.
See general comments about <code>deps</code>
at <a href="${link common-definitions#typical.deps}">Typical attributes defined by
most build rules</a>.
<p>
  This attribute should be used to list other <code>sh_library</code> rules that provide
  interpreted program source code depended on by the code in <code>srcs</code>. The files
  provided by these rules will be present among the <code>runfiles</code> of this target.
</p>
""",
        ),
    },
)
