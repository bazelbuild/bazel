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
import static com.google.devtools.build.lib.analysis.testing.ResolvedToolchainContextSubject.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.ImmutableSetMultimap;
import com.google.devtools.build.lib.analysis.config.ToolchainTypeRequirement;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainException;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContextImpl;
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

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainTypeInfo))
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        testToolchainTypeInfo,
                        Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext).hasToolchainType(testToolchainTypeLabel);
    assertThat(toolchainContext)
        .forToolchainType(testToolchainTypeLabel)
        .getValue("data")
        .isEqualTo("baz");
  }

  @Test
  public void load_mandatory_missing() throws Exception {
    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainTypeInfo))
            .build();

    // Resolve toolchains.
    assertThrows(
        ToolchainException.class,
        () -> ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of()));
  }

  @Test
  public void load_optional_present() throws Exception {
    addOptionalToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(optionalToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(optionalToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(optionalToolchainTypeLabel, optionalToolchainTypeInfo))
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        optionalToolchainTypeInfo,
                        Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext).hasToolchainType(optionalToolchainTypeLabel);
    assertThat(toolchainContext)
        .forToolchainType(optionalToolchainTypeLabel)
        .getValue("data")
        .isEqualTo("baz");
  }

  @Test
  public void load_optional_missing() throws Exception {
    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(optionalToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(optionalToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(optionalToolchainTypeLabel, optionalToolchainTypeInfo))
            .build();

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of());
    assertThat(toolchainContext).isNotNull();

    // Missing optional toolchain type requirement is present.
    assertThat(toolchainContext).hasToolchainType(optionalToolchainTypeLabel);
    // Missing optional toolchain implementation is null.
    assertThat(toolchainContext).forToolchainType(optionalToolchainTypeLabel).isNull();
  }

  @Test
  public void load_mixed() throws Exception {
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType, optionalToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(testToolchainType, optionalToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.<Label, ToolchainTypeInfo>builder()
                    .put(testToolchainTypeLabel, testToolchainTypeInfo)
                    .put(optionalToolchainTypeLabel, optionalToolchainTypeInfo)
                    .build())
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        testToolchainTypeInfo,
                        Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData testToolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//extra:extra_toolchain_linux_impl"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(
            unloadedToolchainContext, "test", ImmutableSet.of(testToolchain));
    assertThat(toolchainContext).isNotNull();

    // Test toolchain is present.
    assertThat(toolchainContext).hasToolchainType(testToolchainTypeLabel);
    assertThat(toolchainContext)
        .forToolchainType(testToolchainTypeLabel)
        .getValue("data")
        .isEqualTo("baz");

    // Missing optional toolchain type requirement is present.
    assertThat(toolchainContext).hasToolchainType(optionalToolchainTypeLabel);
    // Missing optional toolchain implementation is null.
    assertThat(toolchainContext).forToolchainType(optionalToolchainTypeLabel).isNull();
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

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainTypeInfo))
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(testToolchainTypeInfo, Label.parseCanonicalUnchecked("//alias:toolchain"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//alias:toolchain"), targetConfig);

    // Resolve toolchains.
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext).hasToolchainType(testToolchainTypeLabel);
    assertThat(toolchainContext)
        .forToolchainType(testToolchainTypeLabel)
        .getValue("data")
        .isEqualTo("baz");
  }

  @Test
  public void load_notToolchain() throws Exception {
    scratch.file("foo/BUILD", "filegroup(name = 'not_a_toolchain')");

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(testToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(testToolchainTypeLabel, testToolchainTypeInfo))
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        testToolchainTypeInfo,
                        Label.parseCanonicalUnchecked("//foo:not_a_toolchain"))
                    .build())
            .build();

    // Create the prerequisites, which is not actually a valid toolchain.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//foo:not_a_toolchain"), targetConfig);
    assertThrows(
        ToolchainException.class,
        () ->
            ResolvedToolchainContext.load(
                unloadedToolchainContext, "test", ImmutableSet.of(toolchain)));
  }

  @Test
  public void load_withTemplateVariables() throws Exception {
    // Add new toolchain rule that provides template variables.
    Label variableToolchainTypeLabel =
        Label.parseCanonicalUnchecked("//variable:variable_toolchain_type");
    ToolchainTypeRequirement variableToolchainType =
        ToolchainTypeRequirement.create(variableToolchainTypeLabel);
    ToolchainTypeInfo variableToolchainTypeInfo =
        ToolchainTypeInfo.create(variableToolchainTypeLabel);
    scratch.file(
        "variable/variable_toolchain_def.bzl",
        """
        def _impl(ctx):
            value = ctx.attr.value
            toolchain = platform_common.ToolchainInfo()
            template_variables = platform_common.TemplateVariableInfo({"VALUE": value})
            return [toolchain, template_variables]

        variable_toolchain = rule(
            implementation = _impl,
            attrs = {"value": attr.string()},
        )
        """);

    // Create instance of new toolchain and register it.
    scratch.file(
        "variable/BUILD",
        """
        load("//variable:variable_toolchain_def.bzl", "variable_toolchain")

        toolchain_type(name = "variable_toolchain_type")

        variable_toolchain(
            name = "variable_toolchain_impl",
            value = "foo",
        )
        """);

    ToolchainContextKey toolchainContextKey =
        ToolchainContextKey.key()
            .configurationKey(targetConfigKey)
            .toolchainTypes(testToolchainType)
            .build();

    // Create a static UnloadedToolchainContext.
    UnloadedToolchainContext unloadedToolchainContext =
        UnloadedToolchainContextImpl.builder(toolchainContextKey)
            .setExecutionPlatform(linuxPlatform)
            .setTargetPlatform(linuxPlatform)
            .setToolchainTypes(ImmutableSet.of(variableToolchainType))
            .setRequestedLabelToToolchainType(
                ImmutableMap.of(variableToolchainTypeLabel, variableToolchainTypeInfo))
            .setToolchainTypeToResolved(
                ImmutableSetMultimap.<ToolchainTypeInfo, Label>builder()
                    .put(
                        variableToolchainTypeInfo,
                        Label.parseCanonicalUnchecked("//variable:variable_toolchain_impl"))
                    .build())
            .build();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseCanonicalUnchecked("//variable:variable_toolchain_impl"), targetConfig);
    ResolvedToolchainContext toolchainContext =
        ResolvedToolchainContext.load(unloadedToolchainContext, "test", ImmutableSet.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext).hasToolchainType(variableToolchainTypeLabel);
    assertThat(toolchainContext.templateVariableProviders()).hasSize(1);
    assertThat(toolchainContext.templateVariableProviders().get(0).getVariables())
        .containsExactly("VALUE", "foo");
  }
}
