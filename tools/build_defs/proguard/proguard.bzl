# Copyright 2026 The Bazel Authors. All rights reserved.
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

# WARNING:
# https://github.com/bazelbuild/bazel/issues/17713
# .bzl files in this package (tools/build_defs/repo) are evaluated
# in a Starlark environment without "@_builtins" injection, and must not refer
# to symbols associated with build/workspace .bzl files

"""Apply proguard rules to a JAR file."""

load("@rules_java//java:defs.bzl", "java_common")

def _proguard_jar_impl(ctx):
    java_runtime = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
    inputs = ctx.files.srcs + ctx.files.deps + [
        ctx.file.proguard_spec,
        ctx.file._proguard,
    ]
    output = ctx.outputs.out

    args = ctx.actions.args()
    args.add("--java_executable", java_runtime.java_executable_exec_path)
    args.add("--proguard_jar", ctx.file._proguard)
    args.add_joined("--srcs", ctx.files.srcs, join_with = ",")
    args.add_joined("--deps", ctx.files.deps, join_with = ",")
    args.add("--proguard_spec", ctx.file.proguard_spec)
    args.add("--output", output)
    args.add("--timestamp", "1980-01-01 00:00:00")

    ctx.actions.run(
        inputs = inputs,
        mnemonic = "ProguardJar",
        outputs = [output],
        executable = ctx.executable._wrapper,
        arguments = [args],
        tools = java_runtime.files,
    )

    return DefaultInfo(files = depset([output]))

proguard_jar = rule(
    implementation = _proguard_jar_impl,
    attrs = {
        "srcs": attr.label_list(),
        "deps": attr.label_list(),
        "proguard_spec": attr.label(allow_single_file = True),
        "out": attr.output(),
        "_java_runtime": attr.label(
            cfg = "exec",
            default = "@bazel_tools//tools/jdk:current_java_runtime",
            providers = [java_common.JavaRuntimeInfo],
        ),
        "_proguard": attr.label(
            allow_single_file = True,
            cfg = "exec",
            default = ":proguard_private_deploy.jar",
        ),
        "_wrapper": attr.label(
            cfg = "exec",
            default = ":wrapper_private",
            executable = True,
        ),
    },
)
