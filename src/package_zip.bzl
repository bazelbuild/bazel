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

"""Hermetic Starlark rule for building Bazel package.zip archives."""

load("@rules_java//java:defs.bzl", "java_common")

def _package_zip_impl(ctx):
    package_zipper = ctx.file._package_zipper
    java_runtime = ctx.attr._java_runtime[java_common.JavaRuntimeInfo]
    args = ctx.actions.args()
    if ctx.attr.dev_build:
        args.add("--fast")
    args.add(ctx.outputs.out)
    args.add(ctx.file.server_jar)
    args.add(ctx.file.install_base_key)
    args.add_all(ctx.files.srcs)

    ctx.actions.run(
        executable = java_runtime.java_executable_exec_path,
        arguments = ["-jar", package_zipper.path, args],
        inputs = [package_zipper, ctx.file.server_jar, ctx.file.install_base_key] + ctx.files.srcs,
        outputs = [ctx.outputs.out],
        tools = java_runtime.files,
        mnemonic = "PackageZip",
        progress_message = "Building %{output}",
    )

package_zip = rule(
    implementation = _package_zip_impl,
    attrs = {
        "server_jar": attr.label(allow_single_file = True, mandatory = True),
        "install_base_key": attr.label(allow_single_file = True, mandatory = True),
        "srcs": attr.label_list(allow_files = True),
        "out": attr.output(mandatory = True),
        "dev_build": attr.bool(
            default = False,
            doc = "If True, use fast compression (level 1) for developer builds.",
        ),
        "_package_zipper": attr.label(
            default = "//src/java_tools/singlejar/java/com/google/devtools/build/zip:package_zipper_deploy.jar",
            allow_single_file = True,
            cfg = "exec",
        ),
        "_java_runtime": attr.label(
            cfg = "exec",
            default = "@bazel_tools//tools/jdk:current_java_runtime",
            providers = [java_common.JavaRuntimeInfo],
        ),
    },
)
