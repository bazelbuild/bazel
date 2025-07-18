# Copyright 2022 The Bazel Authors. All rights reserved.
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
""" Creates a test for launcher_maker windows tool """

load("@rules_cc//cc:cc_binary.bzl", "cc_binary")

def _impl(ctx):
    launch_info = ctx.actions.args().use_param_file("%s", use_always = True).set_param_file_format("multiline")
    launch_info.add("foo_key=bar")
    foo_list = ["1", "2", "3"]
    launch_info.add_joined(foo_list, join_with = "\t", format_joined = "foo_list=%s")
    launch_info.add_joined([], join_with = "\t", format_joined = "empty_list=%s", omit_if_empty = False)
    output = ctx.actions.declare_file(ctx.label.name + ".exe")
    launcher_artifact = ctx.executable.launcher
    ctx.actions.run(
        executable = ctx.executable._launcher_maker,
        inputs = [launcher_artifact],
        outputs = [output],
        arguments = [launcher_artifact.path, launch_info, output.path],
    )
    return [DefaultInfo(executable = output)]

_launcher_maker_test = rule(
    implementation = _impl,
    attrs = {
        "launcher": attr.label(executable = True, cfg = "target"),
        "_launcher_maker": attr.label(default = ":launcher_maker", executable = True, cfg = "exec"),
    },
    executable = True,
    test = True,
)

def launcher_maker_test(name):
    launcher_exe = name + "_base.exe"
    cc_binary(
        name = launcher_exe,
        srcs = ["launcher_maker_test.cc"],
        deps = [
            "//src/tools/launcher/util",
            "//src/tools/launcher/util:data_parser",
            "@com_google_googletest//:gtest_main",
        ],
    )
    _launcher_maker_test(
        name = name,
        launcher = ":" + launcher_exe,
    )
