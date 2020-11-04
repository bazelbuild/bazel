# Copyright 2020 The Bazel Authors. All rights reserved.
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

"""Provides a rule to run ijar from java_toolchain."""

def _run_ijar(ctx):
    ijar_jar = java_common.run_ijar(
        ctx.actions,
        jar = ctx.file.jar,
        java_toolchain = ctx.attr._java_toolchain[java_common.JavaToolchainInfo],
    )
    return [DefaultInfo(files = depset([ijar_jar]))]

run_ijar = rule(
    implementation = _run_ijar,
    doc = "Runs ijar over the given jar.",
    attrs = {
        "jar": attr.label(mandatory = True, allow_single_file = True),
        "_java_toolchain": attr.label(
            default = "//tools/jdk:current_java_toolchain",
            providers = [java_common.JavaRuntimeInfo],
        ),
    },
)
