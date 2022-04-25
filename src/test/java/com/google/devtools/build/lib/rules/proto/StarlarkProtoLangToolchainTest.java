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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.TransitiveInfoCollection;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.LabelSyntaxException;
import com.google.devtools.build.lib.packages.Provider;
import com.google.devtools.build.lib.packages.StarlarkInfo;
import com.google.devtools.build.lib.packages.StarlarkProvider;
import com.google.devtools.build.lib.testutil.TestConstants;
import net.starlark.java.eval.Starlark;
import net.starlark.java.eval.StarlarkList;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@code proto_lang_toolchain}. */
@RunWith(JUnit4.class)
public class StarlarkProtoLangToolchainTest extends ProtoLangToolchainTest {

  @Override
  @Before
  public void setupStarlarkRule() throws Exception {
    setBuildLanguageOptions("--experimental_builtins_injection_override=+proto_lang_toolchain");
  }

  Provider.Key getStarlarkProtoLangToolchainInfoKey() throws LabelSyntaxException {
    return new StarlarkProvider.Key(
        Label.parseAbsolute("@_builtins//:common/proto/providers.bzl", ImmutableMap.of()),
        "ProtoLangToolchainInfo");
  }

  @SuppressWarnings("unchecked")
  private void validateStarlarkProtoLangToolchain(StarlarkInfo toolchain) throws Exception {
    assertThat(toolchain.getValue("out_replacement_format_flag")).isEqualTo("cmd-line:%s");
    assertThat(toolchain.getValue("plugin_format_flag")).isEqualTo("--plugin=%s");
    assertThat(toolchain.getValue("progress_message")).isEqualTo("Progress Message %{label}");
    assertThat(toolchain.getValue("mnemonic")).isEqualTo("MyMnemonic");
    assertThat(ImmutableList.copyOf((StarlarkList<String>) toolchain.getValue("protoc_opts")))
        .containsExactly("--myflag");
    assertThat(
            ((FilesToRunProvider) toolchain.getValue("plugin"))
                .getExecutable()
                .getRootRelativePathString())
        .isEqualTo("third_party/x/plugin");

    TransitiveInfoCollection runtimes = (TransitiveInfoCollection) toolchain.getValue("runtime");
    assertThat(runtimes.getLabel())
        .isEqualTo(Label.parseAbsolute("//third_party/x:runtime", ImmutableMap.of()));

    Label protoc = Label.parseAbsoluteUnchecked(ProtoConstants.DEFAULT_PROTOC_LABEL);
    assertThat(
            ((FilesToRunProvider) toolchain.getValue("proto_compiler"))
                .getExecutable()
                .prettyPrint())
        .isEqualTo(protoc.toPathFragment().getPathString());
  }

  @Override
  @Test
  public void protoToolchain() throws Exception {
    scratch.file(
        "third_party/x/BUILD",
        "licenses(['unencumbered'])",
        "cc_binary(name = 'plugin', srcs = ['plugin.cc'])",
        "cc_library(name = 'runtime', srcs = ['runtime.cc'])",
        "filegroup(name = 'descriptors', srcs = ['metadata.proto', 'descriptor.proto'])",
        "filegroup(name = 'any', srcs = ['any.proto'])",
        "proto_library(name = 'denied', srcs = [':descriptors', ':any'])");

    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "licenses(['unencumbered'])",
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line:$(OUT)',",
        "    plugin_format_flag = '--plugin=%s',",
        "    plugin = '//third_party/x:plugin',",
        "    runtime = '//third_party/x:runtime',",
        "    progress_message = 'Progress Message %{label}',",
        "    mnemonic = 'MyMnemonic',",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    validateStarlarkProtoLangToolchain(
        (StarlarkInfo)
            getConfiguredTarget("//foo:toolchain").get(getStarlarkProtoLangToolchainInfoKey()));
  }

  @Override
  @Test
  public void protoToolchainBlacklistProtoLibraries() throws Exception {
    scratch.file(
        "third_party/x/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "cc_binary(name = 'plugin', srcs = ['plugin.cc'])",
        "cc_library(name = 'runtime', srcs = ['runtime.cc'])",
        "proto_library(name = 'descriptors', srcs = ['metadata.proto', 'descriptor.proto'])",
        "proto_library(name = 'any', srcs = ['any.proto'], strip_import_prefix = '/third_party')");

    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line:$(OUT)',",
        "    plugin_format_flag = '--plugin=%s',",
        "    plugin = '//third_party/x:plugin',",
        "    runtime = '//third_party/x:runtime',",
        "    progress_message = 'Progress Message %{label}',",
        "    mnemonic = 'MyMnemonic',",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    validateStarlarkProtoLangToolchain(
        (StarlarkInfo)
            getConfiguredTarget("//foo:toolchain").get(getStarlarkProtoLangToolchainInfoKey()));
  }

  @Override
  @Test
  public void protoToolchainBlacklistTransitiveProtos() throws Exception {
    scratch.file(
        "third_party/x/BUILD",
        TestConstants.LOAD_PROTO_LIBRARY,
        "licenses(['unencumbered'])",
        "cc_binary(name = 'plugin', srcs = ['plugin.cc'])",
        "cc_library(name = 'runtime', srcs = ['runtime.cc'])",
        "proto_library(name = 'descriptors', srcs = ['metadata.proto', 'descriptor.proto'])",
        "proto_library(name = 'any', srcs = ['any.proto'], deps = [':descriptors'])");

    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line:$(OUT)',",
        "    plugin_format_flag = '--plugin=%s',",
        "    plugin = '//third_party/x:plugin',",
        "    runtime = '//third_party/x:runtime',",
        "    progress_message = 'Progress Message %{label}',",
        "    mnemonic = 'MyMnemonic',",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    validateStarlarkProtoLangToolchain(
        (StarlarkInfo)
            getConfiguredTarget("//foo:toolchain").get(getStarlarkProtoLangToolchainInfoKey()));
  }

  @Override
  @Test
  public void optionalFieldsAreEmpty() throws Exception {
    scratch.file(
        "foo/BUILD",
        TestConstants.LOAD_PROTO_LANG_TOOLCHAIN,
        "proto_lang_toolchain(",
        "    name = 'toolchain',",
        "    command_line = 'cmd-line:$(OUT)',",
        ")");

    update(ImmutableList.of("//foo:toolchain"), false, 1, true, new EventBus());

    StarlarkInfo toolchain =
        (StarlarkInfo)
            getConfiguredTarget("//foo:toolchain").get(getStarlarkProtoLangToolchainInfoKey());

    assertThat(toolchain.getValue("plugin")).isEqualTo(Starlark.NONE);
    assertThat(toolchain.getValue("runtime")).isEqualTo(Starlark.NONE);
    assertThat(toolchain.getValue("mnemonic")).isEqualTo("GenProto");
  }
}
