# Copyright 2017 The Bazel Authors. All rights reserved.
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

def _java_host_runtime_alias_impl(ctx):
  vars = ctx.attr._host_java_runtime[platform_common.TemplateVariableInfo]
  runtime_info = ctx.attr._host_java_runtime[java_common.JavaRuntimeInfo]
  runtime_toolchain = ctx.attr._host_java_runtime[platform_common.ToolchainInfo]
  return struct(providers=[vars, runtime_info, runtime_toolchain])

java_host_runtime_alias = rule(
    attrs = {
        "_host_java_runtime": attr.label(
            default = Label("//tools/jdk:java_runtime_alias"),
            cfg = "host",
        ),
    },
    implementation = _java_host_runtime_alias_impl,
)
