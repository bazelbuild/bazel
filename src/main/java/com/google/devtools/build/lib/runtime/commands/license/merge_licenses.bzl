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

"""A platform-independent build rule that merges license files."""

def _windows_action(ctx, files):
    cmd = "(FOR %F IN (%SRCS%) DO ((SET X=%F)&ECHO ===== !X:\\=/! =====&TYPE %F&ECHO.&ECHO.)) > %OUT%"
    ctx.actions.run(
        inputs = depset(direct = files),
        outputs = [ctx.outputs.out],
        executable = "cmd.exe",
        arguments = ["/V:ON", "/E:ON", "/Q", "/C", cmd],
        env = {
            "OUT": ctx.outputs.out.path.replace("/", "\\"),
            "SRCS": " ".join([f.path.replace("/", "\\") for f in files]),
        },
    )

def _bash_action(ctx, files):
    cmd = "for f in $SRCS; do echo ===== $f ===== && cat $f && echo && echo ; done > $OUT"
    ctx.actions.run_shell(
        inputs = depset(direct = files),
        outputs = [ctx.outputs.out],
        command = cmd,
        env = {
            "OUT": ctx.outputs.out.path,
            "SRCS": " ".join([f.path for f in files]),
        },
    )

def _impl(ctx):
    files = []
    for src in ctx.files.srcs:
        for substr in ["ASSEMBLY_EXCEPTION", "DISCLAIMER", "LICENSE", "license", "THIRD_PARTY_README"]:
            if substr in src.path:
                files.append(src)
                break
    if not files:
        fail("expected some sources")
    if ctx.attr.is_windows:
        _windows_action(ctx, files)
    else:
        _bash_action(ctx, files)

    return [DefaultInfo(files = depset(direct = [ctx.outputs.out]))]

_merge_licenses = rule(
    implementation = _impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True, mandatory = True),
        "out": attr.output(mandatory = True),
        "is_windows": attr.bool(mandatory = True),
    },
)

def merge_licenses(name, srcs, out, **kwargs):
    _merge_licenses(
        name = name,
        srcs = srcs,
        out = out,
        is_windows = select({
            "@bazel_tools//src/conditions:windows": True,
            "//conditions:default": False,
        }),
        **kwargs
    )
