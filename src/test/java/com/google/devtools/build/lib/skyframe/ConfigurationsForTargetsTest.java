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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData.SPLIT_DEP_ORDERING;

import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolutionHelpers;
import com.google.devtools.build.lib.analysis.PlatformOptions;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.ToolchainCollection;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.ConfigConditions;
import com.google.devtools.build.lib.analysis.platform.PlatformInfo;
import com.google.devtools.build.lib.analysis.producers.DependencyContext;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.toolchains.ToolchainContextKey;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContext;
import com.google.devtools.build.lib.skyframe.toolchains.UnloadedToolchainContextImpl;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import javax.annotation.Nullable;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ConfiguredTargetFunction}'s logic for determining each target's {@link
 * BuildConfigurationValue}.
 *
 * <p>This is essentially an integration test for {@link DependencyResolver#computeDependencies} and
 * {@link DependencyResolutionHelpers}. These methods form the core logic that figures out what a
 * target's deps are, how their configurations should differ from their parent, and how to
 * instantiate those configurations as tangible {@link BuildConfigurationValue} objects.
 *
 * <p>{@link ConfiguredTargetFunction} is a complicated class that does a lot of things. This test
 * focuses purely on the task of determining configurations for deps. So instead of evaluating full
 * {@link ConfiguredTargetFunction} instances, it evaluates a mock {@link SkyFunction} that just
 * wraps the {@link DependencyResolver#computeDependencies} part. This keeps focus tight and
 * integration dependencies narrow.
 *
 * <p>We can't just call {@link DependencyResolver#computeDependencies} directly because that method
 * needs a {@link SkyFunction.Environment} and Blaze's test infrastructure doesn't support direct
 * access to environments.
 */
@RunWith(JUnit4.class)
public final class ConfigurationsForTargetsTest extends AnalysisTestCase {

  private static final Label TARGET_PLATFORM_LABEL =
      Label.parseCanonicalUnchecked("//platform:target");
  private static final Label EXEC_PLATFORM_LABEL = Label.parseCanonicalUnchecked("//platform:exec");

  /**
   * A mock {@link SkyFunction} that just calls {@link DependencyResolver#computeDependencies} and
   * returns its results.
   */
  private static class ComputeDependenciesFunction implements SkyFunction {
    static final SkyFunctionName SKYFUNCTION_NAME =
        SkyFunctionName.createHermetic("CONFIGURED_TARGET_FUNCTION_COMPUTE_DEPENDENCIES");

    private final LateBoundStateProvider stateProvider;

    ComputeDependenciesFunction(LateBoundStateProvider lateBoundStateProvider) {
      this.stateProvider = lateBoundStateProvider;
    }

    /** Returns a {@link SkyKey} for a given <Target, BuildConfigurationValue> pair. */
    private static Key key(Target target, BuildConfigurationValue config) {
      return new Key(new TargetAndConfiguration(target, config));
    }

    private static class Key extends AbstractSkyKey<TargetAndConfiguration> {
      private Key(TargetAndConfiguration arg) {
        super(arg);
      }

      @Override
      public SkyFunctionName functionName() {
        return SKYFUNCTION_NAME;
      }
    }

    /** Returns an {@link OrderedSetMultimap} representing the deps of given target. */
    static final class Value implements SkyValue {
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap;

      Value(OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap) {
        this.depMap = depMap;
      }
    }

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws EvalException, InterruptedException {
      try {
        var targetAndConfiguration = (TargetAndConfiguration) skyKey.argument();
        // Set up the toolchain context so that exec transitions resolve properly.
        var state = DependencyResolver.State.createForTesting(targetAndConfiguration);
        state.dependencyContext =
            DependencyContext.create(
                ToolchainCollection.<UnloadedToolchainContext>builder()
                    .addDefaultContext(
                        UnloadedToolchainContextImpl.builder(
                                ToolchainContextKey.key()
                                    .toolchainTypes(ImmutableSet.of())
                                    .configurationKey(
                                        targetAndConfiguration.getConfiguration().getKey())
                                    .build())
                            .setTargetPlatform(
                                PlatformInfo.builder().setLabel(TARGET_PLATFORM_LABEL).build())
                            .setExecutionPlatform(
                                PlatformInfo.builder().setLabel(EXEC_PLATFORM_LABEL).build())
                            .build())
                    .build(),
                ConfigConditions.EMPTY);
        OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap =
            DependencyResolver.computeDependencies(
                state,
                ConfiguredTargetKey.builder()
                    .setLabel(targetAndConfiguration.getLabel())
                    .setConfiguration(targetAndConfiguration.getConfiguration())
                    .build(),
                /* aspects= */ ImmutableList.of(),
                stateProvider.lateBoundSkyframeBuildView().getStarlarkTransitionCache(),
                /* starlarkTransitionProvider= */ null,
                env,
                env.getListener());
        return env.valuesMissing() ? null : new Value(depMap);
      } catch (RuntimeException e) {
        throw e;
      } catch (Exception e) {
        throw new EvalException(e);
      }
    }

    private static final class EvalException extends SkyFunctionException {
      EvalException(Exception cause) {
        super(cause, Transience.PERSISTENT); // We can generalize the transience if/when needed.
      }
    }

    @Override
    public String extractTag(SkyKey skyKey) {
      return ((TargetAndConfiguration) skyKey.argument()).getLabel().getName();
    }
  }

  /**
   * Provides build state to {@link ComputeDependenciesFunction}. This needs to be late-bound (i.e.
   * we can't just pass the contents directly) because of the way {@link AnalysisTestCase} works:
   * the {@link AnalysisMock} instance that instantiates the function gets created before the rest
   * of the build state. See {@link AnalysisTestCase#createMocks} for details.
   */
  private final class LateBoundStateProvider {
    SkyframeBuildView lateBoundSkyframeBuildView() {
      return skyframeExecutor.getSkyframeBuildView();
    }
  }

  /**
   * An {@link AnalysisMock} that injects {@link ComputeDependenciesFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithComputeDepsFunction extends AnalysisMock.Delegate {
    private final LateBoundStateProvider stateProvider;

    AnalysisMockWithComputeDepsFunction(LateBoundStateProvider stateProvider) {
      super(AnalysisMock.get());
      this.stateProvider = stateProvider;
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(
              ComputeDependenciesFunction.SKYFUNCTION_NAME,
              new ComputeDependenciesFunction(stateProvider))
          .buildOrThrow();
    }
  }

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithComputeDepsFunction(new LateBoundStateProvider());
  }

  /** Returns the configured deps for a given target. */
  private Multimap<DependencyKind, ConfiguredTargetAndData> getConfiguredDeps(
      ConfiguredTarget target) throws Exception {
    String targetLabel = AliasProvider.getDependencyLabel(target).toString();
    SkyKey key = ComputeDependenciesFunction.key(getTarget(targetLabel), getConfiguration(target));
    // Must re-enable analysis for Skyframe functions that create configured targets.
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
    EvaluationResult<ComputeDependenciesFunction.Value> evalResult =
        SkyframeExecutorTestUtils.evaluate(skyframeExecutor, key, /*keepGoing=*/ false, reporter);
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    return evalResult.get(key).depMap;
  }

  /**
   * Returns the configured deps for a given target under the given attribute. Assumes the target
   * uses the target configuration.
   *
   * <p>Throws an exception if the attribute can't be found.
   */
  private ImmutableList<ConfiguredTarget> getConfiguredDeps(String targetLabel, String attrName)
      throws Exception {
    ConfiguredTarget target = Iterables.getOnlyElement(update(targetLabel).getTargetsToBuild());
    ImmutableList<ConfiguredTarget> maybeConfiguredDeps = getConfiguredDeps(target, attrName);
    assertThat(maybeConfiguredDeps).isNotNull();
    return maybeConfiguredDeps;
  }

  /**
   * Returns the configured deps for a given configured target under the given attribute.
   *
   * <p>Returns null if the attribute can't be found.
   */
  @Nullable
  private ImmutableList<ConfiguredTarget> getConfiguredDeps(
      ConfiguredTarget target, String attrName) throws Exception {
    Multimap<DependencyKind, ConfiguredTargetAndData> allDeps = getConfiguredDeps(target);
    for (DependencyKind kind : allDeps.keySet()) {
      Attribute attribute = kind.getAttribute();
      if (attribute.getName().equals(attrName)) {
        return ImmutableList.copyOf(
            Collections2.transform(
                allDeps.get(kind), ConfiguredTargetAndData::getConfiguredTarget));
      }
    }
    return null;
  }

  private ImmutableList<ConfiguredTargetAndData> getConfiguredDepsWithData(
      String targetLabel, String attrName) throws Exception {
    ConfiguredTarget target = Iterables.getOnlyElement(update(targetLabel).getTargetsToBuild());
    ImmutableList<ConfiguredTargetAndData> maybeConfiguredDeps =
        getConfiguredDepsWithData(target, attrName);
    assertThat(maybeConfiguredDeps).isNotNull();
    return maybeConfiguredDeps;
  }

  @Nullable
  private ImmutableList<ConfiguredTargetAndData> getConfiguredDepsWithData(
      ConfiguredTarget target, String attrName) throws Exception {
    Multimap<DependencyKind, ConfiguredTargetAndData> allDeps = getConfiguredDeps(target);
    for (DependencyKind kind : allDeps.keySet()) {
      Attribute attribute = kind.getAttribute();
      if (attribute.getName().equals(attrName)) {
        return ImmutableList.copyOf(allDeps.get(kind));
      }
    }
    return null;
  }

  @Before
  public void setUp() throws Exception {
    scratch.file(
        "platform/BUILD",
        // Add basic target and exec platforms for testing.
        "platform(name = 'target')",
        "platform(name = 'exec')");
  }

  @Test
  public void nullConfiguredDepsHaveExpectedConfigs() throws Exception {
    scratch.file(
        "a/BUILD", "genrule(name = 'gen', srcs = ['gen.in'], cmd = '', outs = ['gen.out'])");
    ConfiguredTarget genIn = Iterables.getOnlyElement(getConfiguredDeps("//a:gen", "srcs"));
    assertThat(getConfiguration(genIn)).isNull();
  }

  @Test
  public void genQueryScopeHasExpectedConfigs() throws Exception {
    scratch.file(
        "p/BUILD",
        "sh_library(name='a')",
        "genquery(name='q', scope=[':a'], expression='deps(//p:a)')");
    ConfiguredTarget target = Iterables.getOnlyElement(update("//p:q").getTargetsToBuild());
    // There are no configured targets for the "scope" attribute.
    @Nullable
    ImmutableList<ConfiguredTarget> configuredScopeDeps = getConfiguredDeps(target, "scope");
    assertThat(configuredScopeDeps).isNull();
  }

  @Test
  public void targetDeps() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_library(name = 'dep1', srcs = ['dep1.cc'])",
        "cc_library(name = 'dep2', srcs = ['dep2.cc'])",
        "cc_binary(name = 'binary', srcs = ['main.cc'], deps = [':dep1', ':dep2'])");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:binary", "deps");
    assertThat(deps).hasSize(2);
    BuildConfigurationValue topLevelConfiguration =
        getConfiguration(Iterables.getOnlyElement(update("//a:binary").getTargetsToBuild()));
    for (ConfiguredTarget dep : deps) {
      assertThat(topLevelConfiguration).isEqualTo(getConfiguration(dep));
    }
  }

  /** Tests dependencies in attribute with exec transition. */
  @Test
  public void execDeps() throws Exception {
    scratch.file(
        "a/exec_rule.bzl",
        "exec_rule = rule(",
        "  implementation = lambda ctx: [],",
        "  attrs = {'tools': attr.label_list(cfg = 'exec')},",
        ")");
    scratch.file(
        "a/BUILD",
        "load('//a:exec_rule.bzl', 'exec_rule')",
        "sh_binary(name = 'exec_tool', srcs = ['exec_tool.sh'])",
        "exec_rule(name = 'gen', tools = [':exec_tool'])");

    ConfiguredTarget toolDep = Iterables.getOnlyElement(getConfiguredDeps("//a:gen", "tools"));
    BuildConfigurationValue toolConfiguration = getConfiguration(toolDep);
    assertThat(toolConfiguration.isToolConfiguration()).isTrue();
    assertThat(toolConfiguration.getOptions().get(PlatformOptions.class).platforms)
        .containsExactly(EXEC_PLATFORM_LABEL);
  }

  @Test
  public void splitDeps() throws Exception {
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "java/a/BUILD",
        "cc_library(name = 'lib', srcs = ['lib.cc'])",
        "android_binary(name='a', manifest = 'AndroidManifest.xml', deps = [':lib'])");
    useConfiguration("--fat_apk_cpu=k8,armeabi-v7a", "--experimental_google_legacy_api");
    ImmutableList<ConfiguredTargetAndData> deps = getConfiguredDepsWithData("//java/a:a", "deps");
    assertThat(deps).hasSize(2);
    ConfiguredTargetAndData dep1 = deps.get(0);
    ConfiguredTargetAndData dep2 = deps.get(1);
    assertThat(ImmutableList.of(dep1.getConfiguration().getCpu(), dep2.getConfiguration().getCpu()))
        .containsExactly("armeabi-v7a", "k8");
    // We don't care what order split deps are listed, but it must be deterministic.
    assertThat(SPLIT_DEP_ORDERING.compare(dep1, dep2)).isLessThan(0);
  }

  /**
   * Ensures that <bold>different</bold> transitions don't trigger false cache hits.
   *
   * <p>This test checks a subtle version of that: if the same Starlark transition applies to two
   * deps, but that transition reads their attributes and their attribute values are different, we
   * need to make sure they're distinctly computed.
   */
  @Test
  public void sameTransitionDifferentParameters() throws Exception {
    scratch.overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//a/...',",
        "    ],",
        ")");
    scratch.file(
        "a/defs.bzl",
        "def _transition_impl(settings, attr):",
        "    return {'//command_line_option:compilation_mode': attr.myattr}",
        "my_transition = transition(",
        "    implementation = _transition_impl,",
        "    inputs = [],",
        "    outputs = ['//command_line_option:compilation_mode'])",
        "def _parent_rule_impl(ctx):",
        "    pass",
        "parent_rule = rule(",
        "    implementation = _parent_rule_impl,",
        "    attrs = {",
        "        'dep1': attr.label(),",
        "        'dep2': attr.label(),",
        "    })",
        "def _child_rule_impl(ctx):",
        "    pass",
        "child_rule = rule(",
        "    implementation = _child_rule_impl,",
        "    cfg = my_transition,",
        "    attrs = {",
        "        'myattr': attr.string(),",
        "        '_allowlist_function_transition': attr.label(",
        "            default = '//tools/allowlists/function_transition_allowlist')",
        "    }",
        ")");
    scratch.file(
        "a/BUILD",
        "load('//a:defs.bzl', 'parent_rule', 'child_rule')",
        "child_rule(",
        "    name = 'child1',",
        "    myattr = 'dbg')", // For this dep, my_transition reads myattr="dbg".
        "child_rule(",
        "    name = 'child2',",
        "    myattr = 'opt')", // For this dep, my_transition reads myattr="opt".
        "parent_rule(",
        "    name = 'buildme',",
        "    dep1 = ':child1',",
        "    dep2 = ':child2')");

    ConfiguredTarget child1 = Iterables.getOnlyElement(getConfiguredDeps("//a:buildme", "dep1"));
    ConfiguredTarget child2 = Iterables.getOnlyElement(getConfiguredDeps("//a:buildme", "dep2"));
    // Check that each dep ends up with a distinct compilation_mode value.
    assertThat(getConfiguration(child1).getCompilationMode()).isEqualTo(CompilationMode.DBG);
    assertThat(getConfiguration(child2).getCompilationMode()).isEqualTo(CompilationMode.OPT);
  }
}
