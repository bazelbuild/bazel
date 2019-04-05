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

package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;
import static com.google.devtools.build.skyframe.EvaluationResultSubjectFactory.assertThatEvaluationResult;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.analysis.ToolchainResolver.NoMatchingPlatformException;
import com.google.devtools.build.lib.analysis.ToolchainResolver.UnloadedToolchainContext;
import com.google.devtools.build.lib.analysis.ToolchainResolver.UnresolvedToolchainsException;
import com.google.devtools.build.lib.analysis.platform.ToolchainTypeInfo;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.rules.platform.ToolchainTestCase;
import com.google.devtools.build.lib.skyframe.BuildConfigurationValue;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.skyframe.ConstraintValueLookupUtil.InvalidConstraintValueException;
import com.google.devtools.build.lib.skyframe.PlatformLookupUtil.InvalidPlatformException;
import com.google.devtools.build.lib.skyframe.ToolchainException;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;
import javax.annotation.Nullable;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for {@link ToolchainResolver}. */
@RunWith(JUnit4.class)
public class ToolchainResolverTest extends ToolchainTestCase {
  /**
   * An {@link AnalysisMock} that injects {@link ResolveToolchainsFunction} into the Skyframe
   * executor.
   */
  private static final class LocalAnalysisMock extends AnalysisMock.Delegate {
    LocalAnalysisMock() {
      super(AnalysisMock.get());
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(RESOLVE_TOOLCHAINS_FUNCTION, new ResolveToolchainsFunction())
          .build();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new LocalAnalysisMock();
  }

  @Test
  public void resolve() throws Exception {
    // This should select platform mac, toolchain extra_toolchain_mac, because platform
    // mac is listed first.
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    addToolchain(
        "extra",
        "extra_toolchain_mac",
        ImmutableList.of("//constraints:mac"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace(
        "register_toolchains('//extra:extra_toolchain_linux', '//extra:extra_toolchain_mac')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    assertThat(unloadedToolchainContext.requiredToolchainTypes())
        .containsExactly(testToolchainType);
    assertThat(unloadedToolchainContext.resolvedToolchainLabels())
        .containsExactly(Label.parseAbsoluteUnchecked("//extra:extra_toolchain_mac_impl"));

    assertThat(unloadedToolchainContext.executionPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:mac"));

    assertThat(unloadedToolchainContext.targetPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_noToolchainType() throws Exception {
    scratch.file("host/BUILD", "platform(name = 'host')");
    rewriteWorkspace("register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--host_platform=//host:host", "--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create("test", ImmutableSet.of(), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    assertThat(unloadedToolchainContext.requiredToolchainTypes()).isEmpty();

    // With no toolchains requested, should fall back to the host platform.
    assertThat(unloadedToolchainContext.executionPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//host:host"));

    assertThat(unloadedToolchainContext.targetPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_noToolchainType_hostNotAvailable() throws Exception {
    scratch.file("host/BUILD", "platform(name = 'host')");
    scratch.file(
        "sample/BUILD",
        "constraint_setting(name='demo')",
        "constraint_value(name = 'demo_a', constraint_setting=':demo')",
        "constraint_value(name = 'demo_b', constraint_setting=':demo')",
        "platform(name = 'sample_a',",
        "  constraint_values = [':demo_a'],",
        ")",
        "platform(name = 'sample_b',",
        "  constraint_values = [':demo_b'],",
        ")");
    rewriteWorkspace(
        "register_execution_platforms('//platforms:mac', '//platforms:linux',",
        "    '//sample:sample_a', '//sample:sample_b')");

    useConfiguration("--host_platform=//host:host", "--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//sample:demo_b")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    assertThat(unloadedToolchainContext.requiredToolchainTypes()).isEmpty();

    assertThat(unloadedToolchainContext.executionPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//sample:sample_b"));

    assertThat(unloadedToolchainContext.targetPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_unavailableToolchainType_single() throws Exception {
    useConfiguration("--host_platform=//platforms:linux", "--platforms=//platforms:mac");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(
                testToolchainTypeLabel, Label.parseAbsoluteUnchecked("//fake/toolchain:type_1")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("no matching toolchains found for types //fake/toolchain:type_1");
  }

  @Test
  public void resolve_unavailableToolchainType_multiple() throws Exception {
    useConfiguration("--host_platform=//platforms:linux", "--platforms=//platforms:mac");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(
                testToolchainTypeLabel,
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_1"),
                Label.parseAbsoluteUnchecked("//fake/toolchain:type_2")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(UnresolvedToolchainsException.class);
    // Only one of the missing types will be reported, so do not check the specific error message.
  }

  @Test
  public void resolve_invalidTargetPlatform_badTarget() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--platforms=//invalid:not_a_platform");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains(
            "//invalid:not_a_platform was referenced as a platform, "
                + "but does not provide PlatformInfo");
  }

  @Test
  public void resolve_invalidTargetPlatform_badPackage() throws Exception {
    scratch.resolve("invalid").delete();
    useConfiguration("--platforms=//invalid:not_a_platform");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("BUILD file not found");
  }

  @Test
  public void resolve_invalidHostPlatform() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--host_platform=//invalid:not_a_platform");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_platform");
  }

  @Test
  public void resolve_invalidExecutionPlatform() throws Exception {
    scratch.file("invalid/BUILD", "filegroup(name = 'not_a_platform')");
    useConfiguration("--extra_execution_platforms=//invalid:not_a_platform");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidPlatformException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//invalid:not_a_platform");
  }

  @Test
  public void resolve_execConstraints() throws Exception {
    // This should select platform linux, toolchain extra_toolchain_linux, due to extra constraints,
    // even though platform mac is registered first.
    addToolchain(
        /* packageName= */ "extra",
        /* toolchainName= */ "extra_toolchain_linux",
        /* execConstraints= */ ImmutableList.of("//constraints:linux"),
        /* targetConstraints= */ ImmutableList.of("//constraints:linux"),
        /* data= */ "baz");
    addToolchain(
        /* packageName= */ "extra",
        /* toolchainName= */ "extra_toolchain_mac",
        /* execConstraints= */ ImmutableList.of("//constraints:mac"),
        /* targetConstraints= */ ImmutableList.of("//constraints:linux"),
        /* data= */ "baz");
    rewriteWorkspace(
        "register_toolchains('//extra:extra_toolchain_linux', '//extra:extra_toolchain_mac')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(testToolchainTypeLabel),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//constraints:linux")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    assertThat(unloadedToolchainContext.requiredToolchainTypes())
        .containsExactly(testToolchainType);
    assertThat(unloadedToolchainContext.resolvedToolchainLabels())
        .containsExactly(Label.parseAbsoluteUnchecked("//extra:extra_toolchain_linux_impl"));

    assertThat(unloadedToolchainContext.executionPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.executionPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));

    assertThat(unloadedToolchainContext.targetPlatform()).isNotNull();
    assertThat(unloadedToolchainContext.targetPlatform().label())
        .isEqualTo(Label.parseAbsoluteUnchecked("//platforms:linux"));
  }

  @Test
  public void resolve_execConstraints_invalid() throws Exception {
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(testToolchainTypeLabel),
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//platforms:linux")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);

    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(InvalidConstraintValueException.class);
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .hasMessageThat()
        .contains("//platforms:linux");
  }

  @Test
  public void resolve_noMatchingPlatform() throws Exception {
    // Write toolchain A, and a toolchain implementing it.
    scratch.appendFile(
        "a/BUILD",
        "toolchain_type(name = 'toolchain_type_A')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type_A',",
        "    exec_compatible_with = ['//constraints:mac'],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "filegroup(name='toolchain_impl')");
    // Write toolchain B, and a toolchain implementing it.
    scratch.appendFile(
        "b/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain_type(name = 'toolchain_type_B')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':toolchain_type_B',",
        "    exec_compatible_with = ['//constraints:linux'],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "filegroup(name='toolchain_impl')");

    rewriteWorkspace(
        "register_toolchains('//a:toolchain', '//b:toolchain')",
        "register_execution_platforms('//platforms:mac', '//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(
                Label.parseAbsoluteUnchecked("//a:toolchain_type_A"),
                Label.parseAbsoluteUnchecked("//b:toolchain_type_B")),
            targetConfigKey);

    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);
    assertThatEvaluationResult(result).hasError();
    assertThatEvaluationResult(result)
        .hasErrorEntryForKeyThat(key)
        .hasExceptionThat()
        .isInstanceOf(NoMatchingPlatformException.class);
  }

  @Test
  public void unloadedToolchainContext_load() throws Exception {
    addToolchain(
        "extra",
        "extra_toolchain_linux",
        ImmutableList.of("//constraints:linux"),
        ImmutableList.of("//constraints:linux"),
        "baz");
    rewriteWorkspace(
        "register_toolchains('//extra:extra_toolchain_linux')",
        "register_execution_platforms('//platforms:linux')");

    useConfiguration("--platforms=//platforms:linux");
    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(testToolchainTypeLabel), targetConfigKey);

    // Create the UnloadedToolchainContext.
    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);
    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//extra:extra_toolchain_linux_impl"), targetConfig);
    ResolvedToolchainContext toolchainContext =
        unloadedToolchainContext.load(ImmutableList.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType)).isNotNull();
    assertThat(toolchainContext.forToolchainType(testToolchainType).hasField("data")).isTrue();
    assertThat(toolchainContext.forToolchainType(testToolchainType).getValue("data"))
        .isEqualTo("baz");
  }

  @Test
  public void unloadedToolchainContext_load_notToolchain() throws Exception {
    scratch.file(
        "foo/BUILD",
        "filegroup(name = 'not_a_toolchain')",
        "toolchain_type(name = 'toolchain_type')",
        "toolchain(",
        "    name = 'test_toolchain',",
        "    toolchain_type = ':toolchain_type',",
        "    toolchain = ':not_a_toolchain')");
    rewriteWorkspace("register_toolchains('//foo:test_toolchain')");

    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test",
            ImmutableSet.of(Label.parseAbsoluteUnchecked("//foo:toolchain_type")),
            targetConfigKey);

    // Create the UnloadedToolchainContext.
    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);
    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    // Create the prerequisites, which is not actually a valid toolchain.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//foo:not_a_toolchain"), targetConfig);
    assertThrows(
        ToolchainException.class, () -> unloadedToolchainContext.load(ImmutableList.of(toolchain)));
  }

  @Test
  public void unloadedToolchainContext_load_withTemplateVariables() throws Exception {
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

    scratch.file("variable/BUILD", "toolchain_type(name = 'variable_toolchain_type')");

    // Create instance of new toolchain and register it.
    scratch.appendFile(
        "BUILD",
        "load('//variable:variable_toolchain_def.bzl', 'variable_toolchain')",
        "toolchain(",
        "    name = 'variable_toolchain',",
        "    toolchain_type = '//variable:variable_toolchain_type',",
        "    exec_compatible_with = [],",
        "    target_compatible_with = [],",
        "    toolchain = ':variable_toolchain_impl')",
        "variable_toolchain(",
        "  name='variable_toolchain_impl',",
        "  value = 'foo')");

    rewriteWorkspace("register_toolchains('//:variable_toolchain')");

    ResolveToolchainsKey key =
        ResolveToolchainsKey.create(
            "test", ImmutableSet.of(variableToolchainTypeLabel), targetConfigKey);

    // Create the UnloadedToolchainContext.
    EvaluationResult<ResolveToolchainsValue> result = createToolchainContextBuilder(key);
    assertThatEvaluationResult(result).hasNoError();
    UnloadedToolchainContext unloadedToolchainContext = result.get(key).unloadedToolchainContext();
    assertThat(unloadedToolchainContext).isNotNull();

    // Create the prerequisites.
    ConfiguredTargetAndData toolchain =
        getConfiguredTargetAndData(
            Label.parseAbsoluteUnchecked("//:variable_toolchain_impl"), targetConfig);
    ResolvedToolchainContext toolchainContext =
        unloadedToolchainContext.load(ImmutableList.of(toolchain));
    assertThat(toolchainContext).isNotNull();
    assertThat(toolchainContext.forToolchainType(variableToolchainType)).isNotNull();
    assertThat(toolchainContext.templateVariableProviders()).hasSize(1);
    assertThat(toolchainContext.templateVariableProviders().get(0).getVariables())
        .containsExactly("VALUE", "foo");
  }

  private static final SkyFunctionName RESOLVE_TOOLCHAINS_FUNCTION =
      SkyFunctionName.createHermetic("RESOLVE_TOOLCHAINS_FUNCTION");

  @AutoValue
  abstract static class ResolveToolchainsKey implements SkyKey {
    @Override
    public SkyFunctionName functionName() {
      return RESOLVE_TOOLCHAINS_FUNCTION;
    }

    abstract String targetDescription();

    abstract ImmutableSet<Label> requiredToolchainTypes();

    abstract ImmutableSet<Label> execConstraintLabels();

    abstract BuildConfigurationValue.Key configurationKey();

    public static ResolveToolchainsKey create(
        String targetDescription,
        Set<Label> requiredToolchains,
        BuildConfigurationValue.Key configurationKey) {
      return create(
          targetDescription,
          requiredToolchains,
          /* execConstraintLabels= */ ImmutableSet.of(),
          configurationKey);
    }

    public static ResolveToolchainsKey create(
        String targetDescription,
        Set<Label> requiredToolchains,
        Set<Label> execConstraintLabels,
        BuildConfigurationValue.Key configurationKey) {
      return new AutoValue_ToolchainResolverTest_ResolveToolchainsKey(
          targetDescription,
          ImmutableSet.copyOf(requiredToolchains),
          ImmutableSet.copyOf(execConstraintLabels),
          configurationKey);
    }
  }

  private EvaluationResult<ResolveToolchainsValue> createToolchainContextBuilder(
      ResolveToolchainsKey key) throws InterruptedException {
    try {
      // Must re-enable analysis for Skyframe functions that create configured targets.
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
      return SkyframeExecutorTestUtils.evaluate(
          skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    } finally {
      skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    }
  }

  @AutoValue
  abstract static class ResolveToolchainsValue implements SkyValue {
    abstract UnloadedToolchainContext unloadedToolchainContext();

    static ResolveToolchainsValue create(UnloadedToolchainContext unloadedToolchainContext) {
      return new AutoValue_ToolchainResolverTest_ResolveToolchainsValue(unloadedToolchainContext);
    }
  }

  private static final class ResolveToolchainsFunction implements SkyFunction {

    @Nullable
    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws SkyFunctionException, InterruptedException {
      ResolveToolchainsKey key = (ResolveToolchainsKey) skyKey;
      ToolchainResolver toolchainResolver =
          new ToolchainResolver(env, key.configurationKey())
              .setTargetDescription(key.targetDescription())
              .setRequiredToolchainTypes(key.requiredToolchainTypes())
              .setExecConstraintLabels(key.execConstraintLabels());

      try {
        UnloadedToolchainContext unloadedToolchainContext = toolchainResolver.resolve();
        if (unloadedToolchainContext == null) {
          return null;
        }
        return ResolveToolchainsValue.create(unloadedToolchainContext);
      } catch (ToolchainException e) {
        throw new ResolveToolchainsFunctionException(e);
      }
    }

    @Nullable
    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  private static class ResolveToolchainsFunctionException extends SkyFunctionException {
    ResolveToolchainsFunctionException(ToolchainException e) {
      super(e, Transience.PERSISTENT);
    }
  }
}
