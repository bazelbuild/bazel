// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import static com.google.devtools.build.lib.analysis.testing.ToolchainCollectionSubject.assertThat;

import com.google.auto.value.AutoValue;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.function.Supplier;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ConfiguredTargetFunction}'s logic for determining each target toolchain context.
 *
 * <p>This is essentially an integration test for {@link
 * ConfiguredTargetFunction#computeUnloadedToolchainContexts}. These methods form the core logic
 * that figures out what a target's toolchain dependencies are.
 *
 * <p>{@link ConfiguredTargetFunction} is a complicated class that does a lot of things. This test
 * focuses purely on the task of toolchain resolution. So instead of evaluating full {@link
 * ConfiguredTargetFunction} instances, it evaluates a mock {@link SkyFunction} that just wraps the
 * {@link ConfiguredTargetFunction#computeUnloadedToolchainContexts} part. This keeps focus tight
 * and integration dependencies narrow.
 *
 * <p>We can't just call {@link ConfiguredTargetFunction#computeUnloadedToolchainContexts} directly
 * because that method needs a {@link SkyFunction.Environment} and Blaze's test infrastructure
 * doesn't support direct access to environments.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ToolchainsForTargetsTest extends AnalysisTestCase {
  /** Returns a {@link SkyKey} for a given <Target, BuildConfiguration> pair. */
  private static Key key(
      TargetAndConfiguration targetAndConfiguration, ConfiguredTargetKey configuredTargetKey) {
    return new AutoValue_ToolchainsForTargetsTest_Key(targetAndConfiguration, configuredTargetKey);
  }

  /** Key class for {@link ComputeUnloadedToolchainContextsFunction}. */
  @AutoValue
  abstract static class Key implements SkyKey {
    abstract TargetAndConfiguration targetAndConfiguration();

    abstract ConfiguredTargetKey configuredTargetKey();

    @Override
    public SkyFunctionName functionName() {
      return ComputeUnloadedToolchainContextsFunction.SKYFUNCTION_NAME;
    }
  }

  /**
   * Returns a {@link ToolchainCollection<UnloadedToolchainContext>} as the result of {@link
   * ConfiguredTargetFunction#computeUnloadedToolchainContexts}.
   */
  @AutoValue
  abstract static class Value implements SkyValue {
    abstract ToolchainCollection<UnloadedToolchainContext> getToolchainCollection();

    static Value create(ToolchainCollection<UnloadedToolchainContext> toolchainCollection) {
      return new AutoValue_ToolchainsForTargetsTest_Value(toolchainCollection);
    }
  }

  /**
   * A mock {@link SkyFunction} that just calls {@link
   * ConfiguredTargetFunction#computeUnloadedToolchainContexts} and returns its results.
   */
  static class ComputeUnloadedToolchainContextsFunction implements SkyFunction {
    static final SkyFunctionName SKYFUNCTION_NAME =
        SkyFunctionName.createHermetic(
            "CONFIGURED_TARGET_FUNCTION_COMPUTE_UNLOADED_TOOLCHAIN_CONTEXTS");

    private final LateBoundStateProvider stateProvider;
    private final Supplier<BuildOptions> buildOptionsSupplier;

    ComputeUnloadedToolchainContextsFunction(
        LateBoundStateProvider lateBoundStateProvider,
        Supplier<BuildOptions> buildOptionsSupplier) {
      this.stateProvider = lateBoundStateProvider;
      this.buildOptionsSupplier = buildOptionsSupplier;
    }

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws ComputeUnloadedToolchainContextsException, InterruptedException {
      try {
        Key key = (Key) skyKey.argument();
        ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
            ConfiguredTargetFunction.computeUnloadedToolchainContexts(
                env,
                stateProvider.lateBoundRuleClassProvider(),
                buildOptionsSupplier.get(),
                key.targetAndConfiguration(),
                key.configuredTargetKey().getToolchainContextKey());
        return env.valuesMissing() ? null : Value.create(toolchainCollection);
      } catch (ToolchainException e) {
        throw new ComputeUnloadedToolchainContextsException(e);
      }
    }

    private static class ComputeUnloadedToolchainContextsException extends SkyFunctionException {
      ComputeUnloadedToolchainContextsException(Exception cause) {
        super(cause, Transience.PERSISTENT); // We can generalize the transience if/when needed.
      }
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return null;
    }
  }

  /**
   * Provides build state to {@link ComputeUnloadedToolchainContextsFunction}. This needs to be
   * late-bound (i.e. we can't just pass the contents directly) because of the way {@link
   * AnalysisTestCase} works: the {@link AnalysisMock} instance that instantiates the function gets
   * created before the rest of the build state. See {@link AnalysisTestCase#createMocks} for
   * details.
   */
  private class LateBoundStateProvider {
    RuleClassProvider lateBoundRuleClassProvider() {
      return ruleClassProvider;
    }
  }

  /**
   * An {@link AnalysisMock} that injects {@link ComputeUnloadedToolchainContextsFunction} into the
   * Skyframe executor.
   */
  private static final class AnalysisMockWithComputeDepsFunction extends AnalysisMock.Delegate {
    private final LateBoundStateProvider stateProvider;
    private final Supplier<BuildOptions> defaultBuildOptions;

    AnalysisMockWithComputeDepsFunction(
        AnalysisMock parent,
        LateBoundStateProvider stateProvider,
        Supplier<BuildOptions> defaultBuildOptions) {
      super(parent);
      this.stateProvider = stateProvider;
      this.defaultBuildOptions = defaultBuildOptions;
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(
              ComputeUnloadedToolchainContextsFunction.SKYFUNCTION_NAME,
              new ComputeUnloadedToolchainContextsFunction(stateProvider, defaultBuildOptions))
          .build();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithComputeDepsFunction(
        super.getAnalysisMock(),
        new LateBoundStateProvider(),
        () -> skyframeExecutor.getDefaultBuildOptions());
  }

  public ToolchainCollection<UnloadedToolchainContext> getToolchainCollection(
      ConfiguredTarget configuredTarget, ConfiguredTargetKey configuredTargetKey)
      throws InterruptedException {
    String targetLabel = configuredTarget.getOriginalLabel().toString();
    SkyKey key =
        key(
            new TargetAndConfiguration(getTarget(targetLabel), getConfiguration(configuredTarget)),
            configuredTargetKey);
    // Analysis phase ended after the update() call in getToolchainCollection. We must re-enable
    // analysis so we can call ConfiguredTargetFunction again without raising an error.
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
    Object evalResult =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    // Test call has finished, to reset the state.
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    @SuppressWarnings("unchecked")
    SkyValue value = ((EvaluationResult<Value>) evalResult).get(key);
    return ((Value) value).getToolchainCollection();
  }

  public ToolchainCollection<UnloadedToolchainContext> getToolchainCollection(String targetLabel)
      throws Exception {
    ConfiguredTarget target = Iterables.getOnlyElement(update(targetLabel).getTargetsToBuild());
    return getToolchainCollection(
        target,
        ConfiguredTargetKey.builder()
            .setLabel(target.getOriginalLabel())
            .setConfigurationKey(target.getConfigurationKey())
            .build());
  }

  @Before
  public void createToolchains() throws Exception {
    scratch.appendFile("WORKSPACE", "register_toolchains('//toolchains:all')");

    scratch.file(
        "toolchain/toolchain_def.bzl",
        "def _impl(ctx):",
        "  toolchain = platform_common.ToolchainInfo(",
        "      data = ctx.attr.data)",
        "  return [toolchain]",
        "test_toolchain = rule(",
        "    implementation = _impl,",
        "    attrs = {",
        "       'data': attr.string()})");

    scratch.file("toolchain/BUILD", "toolchain_type(name = 'test_toolchain')");

    scratch.appendFile(
        "toolchains/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain(",
        "    name = 'toolchain_1',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = [],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_1_impl')",
        "test_toolchain(",
        "  name='toolchain_1_impl',",
        "  data = 'foo')",
        "toolchain(",
        "    name = 'toolchain_2',",
        "    toolchain_type = '//toolchain:test_toolchain',",
        "    exec_compatible_with = [],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_2_impl')",
        "test_toolchain(",
        "    name='toolchain_2_impl',",
        "    data = 'bar')");

    scratch.appendFile(
        "toolchain/rule.bzl",
        "def _impl(ctx):",
        "    data = ctx.toolchains['//toolchain:test_toolchain'].data",
        "    return []",
        "my_rule = rule(",
        "    implementation = _impl,",
        "    toolchains = ['//toolchain:test_toolchain'],",
        ")");
  }

  // actual tests
  @Test
  public void basicToolchains() throws Exception {
    scratch.file("a/BUILD", "load('//toolchain:rule.bzl', 'my_rule')", "my_rule(name = 'a')");

    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection("//a");
    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasToolchainType("//toolchain:test_toolchain");
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasResolvedToolchain("//toolchains:toolchain_1_impl");
  }

  @Test
  public void execPlatform() throws Exception {
    // Add some platforms and custom constraints.
    scratch.file("platforms/BUILD", "platform(name = 'local_platform_a')");

    // Test normal resolution, and with a per-target exec constraint.
    scratch.file("a/BUILD", "load('//toolchain:rule.bzl', 'my_rule')", "my_rule(name = 'a')");

    useConfiguration("--extra_execution_platforms=//platforms:local_platform_a");

    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection("//a");
    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        // First execution platform will be used.
        .hasExecutionPlatform("//platforms:local_platform_a");
  }

  @Test
  public void execPlatform_withExecConstraint() throws Exception {
    // Add some platforms and custom constraints.
    scratch.file(
        "platforms/BUILD",
        "constraint_setting(name = 'local_setting')",
        "constraint_value(name = 'local_value_a', constraint_setting = ':local_setting')",
        "constraint_value(name = 'local_value_b', constraint_setting = ':local_setting')",
        "platform(name = 'local_platform_a',",
        "    constraint_values = [':local_value_a'],",
        ")",
        "platform(name = 'local_platform_b',",
        "    constraint_values = [':local_value_b'],",
        ")");

    // Test normal resolution, and with a per-target exec constraint.
    scratch.file(
        "a/BUILD",
        "load('//toolchain:rule.bzl', 'my_rule')",
        "my_rule(name = 'a',",
        "    exec_compatible_with = ['//platforms:local_value_b'],",
        ")");

    useConfiguration(
        "--extra_execution_platforms=//platforms:local_platform_a,//platforms:local_platform_b");

    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection("//a");
    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        // Exec constraint forces the use of this exec platform.
        .hasExecutionPlatform("//platforms:local_platform_b");
  }

  @Test
  public void execGroups_named() throws Exception {
    // Write a rule with exec groups.
    scratch.appendFile(
        "toolchain/exec_group_rule.bzl",
        "def _impl(ctx):",
        "    pass",
        "my_exec_group_rule = rule(",
        "    implementation = _impl,",
        "    exec_groups = {",
        "        'temp': exec_group(",
        "             toolchains = ['//toolchain:test_toolchain'],",
        "         ),",
        "    },",
        ")");

    scratch.file(
        "a/BUILD",
        "load('//toolchain:exec_group_rule.bzl', 'my_exec_group_rule')",
        "my_exec_group_rule(name = 'a')");

    useConfiguration("--experimental_exec_groups");
    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection("//a");
    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection).defaultToolchainContext().toolchainTypeLabels().isEmpty();
    assertThat(toolchainCollection).defaultToolchainContext().resolvedToolchainLabels().isEmpty();

    assertThat(toolchainCollection).hasExecGroup("temp");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasToolchainType("//toolchain:test_toolchain");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasResolvedToolchain("//toolchains:toolchain_1_impl");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasToolchainType("//toolchain:test_toolchain");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasResolvedToolchain("//toolchains:toolchain_1_impl");
  }

  @Test
  public void execGroups_defaultAndNamed() throws Exception {
    // Add another toolchain type.
    scratch.appendFile(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain_type(name = 'extra_toolchain')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = ':extra_toolchain',",
        "    exec_compatible_with = [],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "test_toolchain(",
        "    name='toolchain_impl',",
        "    data = 'foo')");

    // Write a rule with exec groups.
    scratch.appendFile(
        "toolchain/exec_group_rule.bzl",
        "def _impl(ctx):",
        "    pass",
        "my_exec_group_rule = rule(",
        "    implementation = _impl,",
        "    toolchains = ['//extra:extra_toolchain'],",
        "    exec_groups = {",
        "        'temp': exec_group(",
        "             toolchains = ['//toolchain:test_toolchain'],",
        "         ),",
        "    },",
        ")");

    scratch.file(
        "a/BUILD",
        "load('//toolchain:exec_group_rule.bzl', 'my_exec_group_rule')",
        "my_exec_group_rule(name = 'a')");

    useConfiguration("--experimental_exec_groups", "--extra_toolchains=//extra:toolchain");
    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection("//a");
    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasToolchainType("//extra:extra_toolchain");
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasResolvedToolchain("//extra:toolchain_impl");

    assertThat(toolchainCollection).hasExecGroup("temp");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasToolchainType("//toolchain:test_toolchain");
    assertThat(toolchainCollection)
        .execGroup("temp")
        .hasResolvedToolchain("//toolchains:toolchain_1_impl");
  }

  @Test
  public void keepParentToolchainContext() throws Exception {
    scratch.file(
        "extra/BUILD",
        "load('//toolchain:toolchain_def.bzl', 'test_toolchain')",
        "toolchain_type(name = 'extra_toolchain')",
        "toolchain(",
        "    name = 'toolchain',",
        "    toolchain_type = '//extra:extra_toolchain',",
        "    exec_compatible_with = [],",
        "    target_compatible_with = [],",
        "    toolchain = ':toolchain_impl')",
        "test_toolchain(",
        "    name='toolchain_impl',",
        "    data = 'foo')");
    scratch.file("a/BUILD", "load('//toolchain:rule.bzl', 'my_rule')", "my_rule(name = 'a')");

    useConfiguration("--extra_toolchains=//extra:toolchain");
    ConfiguredTarget target = Iterables.getOnlyElement(update("//a").getTargetsToBuild());
    ToolchainCollection<UnloadedToolchainContext> toolchainCollection =
        getToolchainCollection(
            target,
            ConfiguredTargetKey.builder()
                .setLabel(target.getOriginalLabel())
                .setConfigurationKey(target.getConfigurationKey())
                .setToolchainContextKey(
                    ToolchainContextKey.key()
                        .configurationKey(target.getConfigurationKey())
                        .requiredToolchainTypeLabels(
                            Label.parseAbsoluteUnchecked("//extra:extra_toolchain"))
                        .build())
                .build());

    assertThat(toolchainCollection).isNotNull();
    assertThat(toolchainCollection).hasDefaultExecGroup();
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasToolchainType("//extra:extra_toolchain");
    assertThat(toolchainCollection)
        .defaultToolchainContext()
        .hasResolvedToolchain("//extra:toolchain_impl");
  }
}
