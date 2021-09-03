// Copyright 2021 The Bazel Authors. All rights reserved.
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

import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code ProtoToolchainInfo}. */
@RunWith(JUnit4.class)
public class ProtoToolchainInfoTest extends BuildViewTestCase {
  @Before
  public void setUp() throws Exception {
    scratch.file(
        "proto/toolchain.bzl",
        "def _impl(ctx):",
        "  return ProtoToolchainInfo(",
        "      compiler = ctx.attr.compiler.files_to_run,",
        "      compiler_options = ctx.attr.compiler_options,",
        "  )",
        "proto_toolchain = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'compiler': attr.label(executable=True, cfg='exec'),",
        "    'compiler_options': attr.string_list(),",
        "  },",
        ")");
    scratch.file(
        "proto/BUILD",
        "load(':toolchain.bzl', 'proto_toolchain')",
        "cc_binary(",
        "  name = 'compiler',",
        ")");
  }

  @Test
  public void testStarlarkApi() throws Exception {
    scratch.file(
        "foo/BUILD",
        "load('//proto:toolchain.bzl', 'proto_toolchain')",
        "proto_toolchain(",
        "  name = 'toolchain',",
        "  compiler = '//proto:compiler',",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//foo:toolchain");
    ProtoToolchainInfo protoToolchain = target.get(ProtoToolchainInfo.PROVIDER);
    FilesToRunProvider compiler = protoToolchain.getCompiler();
    assertThat(compiler.getExecutable().getOwner().toString()).isEqualTo("//proto:compiler");
    assertThat(protoToolchain.getCompilerOptions()).isEmpty();
  }

  @Test
  public void testStarlarkApi_withCompilerOptions() throws Exception {
    scratch.file(
        "foo/BUILD",
        "load('//proto:toolchain.bzl', 'proto_toolchain')",
        "proto_toolchain(",
        "  name = 'toolchain',",
        "  compiler = '//proto:compiler',",
        "  compiler_options = ['--foo', '--bar'],",
        ")");

    ConfiguredTarget target = getConfiguredTarget("//foo:toolchain");
    ProtoToolchainInfo protoToolchain = target.get(ProtoToolchainInfo.PROVIDER);
    FilesToRunProvider compiler = protoToolchain.getCompiler();
    assertThat(compiler.getExecutable().getOwner().toString()).isEqualTo("//proto:compiler");
    assertThat(protoToolchain.getCompilerOptions()).containsExactly("--foo", "--bar").inOrder();
  }
}
