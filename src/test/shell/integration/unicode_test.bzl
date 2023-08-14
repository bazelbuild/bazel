# Copyright 2023 The Bazel Authors. All rights reserved.
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

"""Rule implementations exercised in unicode_test"""

def _run_executable_rule_impl(ctx):
    out = ctx.outputs.out
    ctx.actions.run(
        executable = ctx.executable.executable,
        arguments = [out.path] + ctx.attr.extra_arguments,
        outputs = [out],
    )
    return [DefaultInfo(files = depset([out]))]

run_executable_rule = rule(
    implementation = _run_executable_rule_impl,
    doc = "Runs `executable` via ctx.actions.run() with `out` as the first argument and `extra_arguments` as remaining arguments",
    attrs = {
        "executable": attr.label(allow_single_file = True, executable = True, cfg = "exec"),
        "out": attr.output(),
        "extra_arguments": attr.string_list(),
    },
)

def _write_file_rule_impl(ctx):
    out = ctx.outputs.out
    ctx.actions.write(
        output = out,
        content = ctx.attr.content,
        is_executable = ctx.attr.is_executable,
    )
    return [DefaultInfo(files = depset([out]))]

write_file_rule = rule(
    implementation = _write_file_rule_impl,
    doc = "Writes `content` to `out` via ctx.actions.write()",
    attrs = {
        "content": attr.string(),
        "out": attr.output(),
        "is_executable": attr.bool(),
    },
)

def _run_executable_with_param_file_impl(ctx):
    args = ctx.actions.args()
    args.use_param_file("%s", use_always = True)
    args.add(ctx.attr.content)
    ctx.actions.run(
        inputs = [],
        outputs = [ctx.outputs.out],
        arguments = [args, ctx.outputs.out.path],
        executable = ctx.executable.executable,
    )

run_executable_with_param_file_rule = rule(
    implementation = _run_executable_with_param_file_impl,
    doc = "Writes `content` to a param file and passes the file to the executable",
    attrs = {
        "out": attr.output(mandatory = True),
        "content": attr.string(mandatory = True),
        "executable": attr.label(
            allow_files = True,
            executable = True,
            cfg = "exec",
        ),
    },
)
