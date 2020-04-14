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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableBiMap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContext;
import com.google.devtools.build.lib.skyframe.UnloadedToolchainContextImpl;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ResolvedToolchainContext}. */
@RunWith(JUnit4.class)
public class ResolvedToolchainContextTest extends ToolchainTestCase {

  @Test
  public void load() throws Exception {
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder()
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setRequiredToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainType))
            .setToolchainTypeToResolved(
                ImmutableBiMap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        testToolchainType,
                        Label.parseAbsoluteUnchecked("//extra:extra_toolchain_linux_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//extra:extra_toolchain_linux_impl"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(
            toolchain.getTarget().getPackage().getRepositoryMapping(),
            unloadedToolchainContext,
            "test",
            ImmutableList.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType)).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType).getValue("data"))
        .isEqualTo("baz");
  }

  @Test
  public void load_aliasedToolchain() throws Exception {
    scratch.file(
        "alias/BUILD", "alias(name = 'toolchain', actual = '//extra:extra_toolchain_linux_impl')");
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder()
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setRequiredToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainType))
            .setToolchainTypeToResolved(
                ImmutableBiMap.<ToolchainTypeInfo, Label>builder()
                    .put(testToolchainType, Label.parseAbsoluteUnchecked("//alias:toolchain"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(Label.parseAbsoluteUnchecked("//alias:toolchain"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(
            toolchain.getTarget().getPackage().getRepositoryMapping(),
            unloadedToolchainContext,
            "test",
            ImmutableList.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType)).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType).getValue("data"))
        .isEqualTo("baz");
  }

  @Test
  public void load_notToolchain() throws Exception {
    scratch.file("foo/BUILD", "filegroup(name = 'not_a_toolchain')");

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder()
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setRequiredToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainType))
            .setToolchainTypeToResolved(
                ImmutableBiMap.<ToolchainTypeInfo, Label>builder()
                    .put(testToolchainType, Label.parseAbsoluteUnchecked("//foo:not_a_toolchain"))
                    .build())
            .build();

    // Create the prerequisites, which is not actually a valid toolchain.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//foo:not_a_toolchain"), targetConfig);
    assertThrows(
        ToolchainException.class,
        () ->
            ResolvedToolchainContext.load(
                toolchain.getTarget().getPackage().getRepositoryMapping(),
                unloadedToolchainContext,
                "test",
                ImmutableList.of(toolchain)));
  }

  @Test
  public void load_withTemplateVariables() throws Exception {
    // Add new toolchain rule that provides template variables.
    Label variableToolchainTypeLabel =
        Label.parseAbsoluteUnchecked("//variable:variable_toolchain_type");
    ToolchainTypeInfo variableToolchainType = ToolchainTypeInfo.create(variableToolchainTypeLabel);
    scratch.file(
        "variable/variable_toolchain_def.bzl",
        "def _impl(ctx):",
        "  value = ctx.attr.value",
        "  toolchain = platform_common.ToolchainInfo()",
        "  template_variables = platform_common.TemplateVariableInfo({'VALUE': value})",
        "  return [toolchain, template_variables]",
        "variable_toolchain = rule(",
        "    implementation = _impl,",
        "    attrs = {'value': attr.string()})");

    // Create instance of new toolchain and register it.
    scratch.file(
        "variable/BUILD",
        "load('//variable:variable_toolchain_def.bzl', 'variable_toolchain')",
        "toolchain_type(name = 'variable_toolchain_type')",
        "variable_toolchain(",
        "  name='variable_toolchain_impl',",
        "  value = 'foo')");

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder()
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setRequiredToolchainTypes(ImmutableSet.of(variableToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(variableToolchainTypeLabel, variableToolchainType))
            .setToolchainTypeToResolved(
                ImmutableBiMap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        variableToolchainType,
                        Label.parseAbsoluteUnchecked("//variable:variable_toolchain_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//variable:variable_toolchain_impl"), targetConfig);
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(
            toolchain.getTarget().getPackage().getRepositoryMapping(),
            unloadedToolchainContext,
            "test",
            ImmutableList.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext.forToolchainType(variableToolchainType)).isNotNull();
    assertThat(toolchainContext.templateVariableProviders()).hasSize(1);
    assertThat(toolchainContext.templateVariableProviders().get(0).getVariables())
        .containsExactly("VALUE", "foo");
  }
}
