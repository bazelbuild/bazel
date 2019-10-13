// Copyright 2018 The Bazel Authors. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package com.google.devtools.build.lib.skylarkbuildapi;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.skylarkinterface.SkylarkCallable;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModule;
import com.google.devtools.build.lib.skylarkinterface.SkylarkModuleCategory;

/**
 * A configuration fragment representing protocol buffers.
 */
@SkylarkModule(
    name = "proto",
    category = SkylarkModuleCategory.CONFIGURATION_FRAGMENT,
    doc = "A configuration fragment representing protocol buffers. "
              + "<p><b>Do not use these fields directly<b>, they are considered an implementation "
              + "detail and will be removed after migrating Protobuf rules to Starlark.</p>"
              // TODO(yannic): Link to generated docs of `proto_toolchain`.
              // https://github.com/bazelbuild/bazel/issues/9203
              + "<p>Instead, you can access them through proto_toolchain</p>"
              + "<p>pre class=\"language-python\">\n"
              + "def _my_rule_impl(ctx):\n"
              + "    proto_toolchain = ctx.toolchains[\"@rules_proto//proto:toolchain\"]\n"
              + "\n"
              + "    # Contains the protoc binary, as specified by `--proto_compiler`.\n"
              + "    protoc = proto_toolchain.compiler\n"
              + "\n"
              + "    # Contains extra args to pass to protoc, as specified by `--protocopt`.\n"
              + "    compiler_options = proto_toolchain.compiler_options\n"
              + "\n"
              + "    # Contains the strict-dependency mode to use for Protocol Buffers\n"
              + "    # (i.e. 'OFF`, `WARN`, `ERROR`), as specified by `--strict_proto_deps`.\n"
              + "    strict_deps = proto_toolchain.strict_deps\n"
              + "\n"
              + "my_rule = rule(\n"
              + "    implementation = _my_rule_impl,\n"
              + "    attrs = {},\n"
              + "    toolchains = [\n"
              + "        \"@rules_proto//proto:toolchain\",\n"
              + "    ],\n"
              + ")\n"
              + "</pre></p>"
)
public interface ProtoConfigurationApi {
  @SkylarkCallable(
      // Must match the value of `_protocopt_key`
      // in `@rules_proto//proto/private/rules:proto_toolchain.bzl`.
      name = "protocopt_do_not_use_or_we_will_break_you_without_mercy",
      doc = "Exposes the value of `--protocopt`."
                + "<p><b>Do not use this field directly</b>, its only purpose is to help with "
                + "migration of Protobuf rules to Starlark.</p>"
                + "<p>Instead, you can access the value through proto_toolchain</p>"
                + "<p>pre class=\"language-python\">\n"
                + "def _my_rule_impl(ctx):"
                + "    proto_toolchain = ctx.toolchains[\"@rules_proto//proto:toolchain\"]\n"
                + "    compiler_options = proto_toolchain.compiler_options\n"
                + "</pre></p>",
      structField = true)
  ImmutableList<String> protocOpts();

  @SkylarkCallable(
      // Must match the value of `_strict_deps_key`
      // in `@rules_proto//proto/private/rules:proto_toolchain.bzl`.
      name = "strict_deps_do_not_use_or_we_will_break_you_without_mercy",
      doc = "Exposes the value of `--strict_proto_deps`."
                + "<p><b>Do not use this field directly</b>, its only purpose is to help with "
                + "migration of Protobuf rules to Starlark.</p>"
                + "<p>Instead, you can access the value through proto_toolchain</p>"
                + "<p>pre class=\"language-python\">\n"
                + "def _my_rule_impl(ctx):"
                + "    proto_toolchain = ctx.toolchains[\"@rules_proto//proto:toolchain\"]\n"
                + "    strict_deps = proto_toolchain.strict_deps\n"
                + "</pre></p>",
      structField = true)
  String starlarkStrictDeps();
}
