// Copyright 2019 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.rules.proto;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.getFirstArtifactEndingWith;

import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProtoConfigurationTest extends BuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    scratch.file("proto/BUILD", "cc_binary(name='compiler')");
    scratch.file(
        "proto/defs.bzl",
        "def check_proto_common():",
        "    if not hasattr(proto_common,",
        "                   'has_protoc_do_not_use_or_we_will_break_you_without_mercy'):",
        "        fail('missing has_protoc_... attribute on proto_common')",
        "check_proto_common()",
        "",
        "def _echo_protocopt_impl(ctx):",
        "    opts = ctx.fragments.proto.protocopt_do_not_use_or_we_will_break_you_without_mercy",
        "    f = ctx.actions.declare_file(ctx.attr.name)",
        "    ctx.actions.run(executable='echo',outputs=[f], arguments=opts)",
        "    return [DefaultInfo(files=depset([f]))]",
        "echo_protocopt = rule(_echo_protocopt_impl, fragments=['proto'])",
        "",
        "def _echo_strict_deps_impl(ctx):",
        "    s = ctx.fragments.proto.strict_deps_do_not_use_or_we_will_break_you_without_mercy",
        "    f = ctx.actions.declare_file(ctx.attr.name)",
        "    ctx.actions.run(executable='echo',outputs=[f], arguments=[s])",
        "    return [DefaultInfo(files=depset([f]))]",
        "echo_strict_deps = rule(_echo_strict_deps_impl, fragments=['proto'])",
        "",
        "def _echo_proto_compiler_impl(ctx):",
        "    return [DefaultInfo(files=depset([ctx.executable._compiler]))]",
        "_protoc_key = 'protoc_do_not_use_or_we_will_break_you_without_mercy'",
        "echo_proto_compiler = rule(",
        "    implementation = _echo_proto_compiler_impl,",
        "    attrs = {",
        "        '_compiler': attr.label(",
        "            executable = True,",
        "            cfg = 'host',",
        "            default = configuration_field('proto', _protoc_key),",
        "        ),",
        "    },",
        "    fragments=['proto']",
        ")");
  }

  private Artifact getArtifact(String target, String file) throws Exception {
    return getFirstArtifactEndingWith(getFilesToBuild(getConfiguredTarget(target)), file);
  }

  @Test
  public void readProtocopt() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'echo_protocopt')",
        "echo_protocopt(name = 'x')");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .isEmpty();

    useConfiguration("--protocopt=--foo", "--protocopt=--bar=10");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("--foo", "--bar=10");
  }

  @Test
  public void readStrictDeps() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'echo_strict_deps')",
        "echo_strict_deps(name = 'x')");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("ERROR");

    useConfiguration("--strict_proto_deps=OFF");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("OFF");

    useConfiguration("--strict_proto_deps=DEFAULT");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("OFF");

    useConfiguration("--strict_proto_deps=WARN");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("WARN");

    useConfiguration("--strict_proto_deps=ERROR");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("ERROR");

    useConfiguration("--strict_proto_deps=DEFAULT");
    assertThat(getGeneratingSpawnAction(getArtifact("//x", "x")).getRemainingArguments())
        .containsExactly("OFF");
  }

  @Test
  public void readProtoCompiler() throws Exception {
    scratch.file(
        "x/BUILD",
        "load('//proto:defs.bzl', 'echo_proto_compiler')",
        "echo_proto_compiler(name = 'x')");

    useConfiguration("--proto_compiler=//proto:compiler");
    assertThat(getArtifact("//x", "compiler").getRootRelativePathString())
        .startsWith("proto/compiler");  // Ends with `.exe` on Windows.
  }
}
