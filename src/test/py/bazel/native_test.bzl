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

"""For testing only.

This module exports two rules:
- bat_test is a test rule. It writes its executable (a .bat file) with
  user-defined content.
- exe_test is a test rule. It copies a native executable and uses that
  as its own.
"""

def _bat_test_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + ".bat")
    ctx.actions.write(
        output = out,
        content = "\r\n".join(ctx.attr.content),
        is_executable = True,
    )
    return [DefaultInfo(executable = out)]

bat_test = rule(
    implementation = _bat_test_impl,
    test = True,
    attrs = {"content": attr.string_list()},
)

def _exe_test_impl(ctx):
    out = ctx.actions.declare_file(ctx.label.name + "." + ctx.file.src.extension)
    ctx.actions.run(
        tools = [ctx.file.src],
        outputs = [out],
        executable = "cmd.exe",
        arguments = ["/C", "copy /Y %IN% %OUT%"],
        env = {
            "IN": ctx.file.src.path.replace("/", "\\"),
            "OUT": out.path.replace("/", "\\"),
        },
    )
    return [DefaultInfo(executable = out)]

exe_test = rule(
    implementation = _exe_test_impl,
    test = True,
    attrs = {
        "src": attr.label(
            allow_single_file = True,
            cfg = "host",
            executable = True,
        ),
    },
)
