// Copyright 2016 The Bazel Authors. All rights reserved.
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
import static com.google.devtools.build.lib.actions.util.ActionsTestUtil.prettyArtifactNames;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public class ProtoLangToolchainTest extends BuildViewTestCase {
  @Test
  public void protoToolchain() throws Exception {
    scratch.file(
        "x/BUILD",
        "cc_binary(name = 'plugin', srcs = ['plugin.cc'])",
        "cc_library(name = 'runtime', srcs = ['runtime.cc'])",
        "filegroup(name = 'descriptors', srcs = ['metadata.proto', 'descriptor.proto'])",
        "filegroup(name = 'any', srcs = ['any.proto'])");

    scratch.file(
        "foo/BUILD",
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line',",
        "    plugin = '//x:plugin',",
        "    runtime = '//x:runtime',",
        "    blacklisted_protos = ['//x:descriptors', '//x:any']",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    ProtoLangToolchainProvider toolchain =
        getConfiguredTarget("//foo:toolchain").getProvider(ProtoLangToolchainProvider.class);

    assertThat(toolchain.commandLine()).isEqualTo("cmd-line");
    assertThat(toolchain.pluginExecutable().getExecutable().getRootRelativePathString())
        .isEqualTo("x/plugin");

    TransitiveInfoCollection runtimes = toolchain.runtime();
    assertThat(runtimes.getLabel())
        .isEqualTo(Label.parseAbsolute("//x:runtime", ImmutableMap.of()));

    assertThat(prettyArtifactNames(toolchain.blacklistedProtos()))
        .containsExactly("x/metadata.proto", "x/descriptor.proto", "x/any.proto");
  }

  @Test
  public void optionalFieldsAreEmpty() throws Exception {
    scratch.file(
        "foo/BUILD",
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line',",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    ProtoLangToolchainProvider toolchain =
        getConfiguredTarget("//foo:toolchain").getProvider(ProtoLangToolchainProvider.class);

    assertThat(toolchain.pluginExecutable()).isNull();
    assertThat(toolchain.runtime()).isNull();
    assertThat(toolchain.blacklistedProtos()).isEmpty();
  }
}
