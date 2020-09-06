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

import com.google.common.base.Supplier;
import com.google.common.collect.Collections2;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multimap;
import com.google.devtools.build.lib.analysis.AliasProvider;
import com.google.devtools.build.lib.analysis.BlazeDirectories;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.DependencyKind;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.CompilationMode;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.packages.Aspect;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Package;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.skyframe.util.SkyframeExecutorTestUtils;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.OrderedSetMultimap;
import com.google.devtools.build.skyframe.AbstractSkyKey;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests {@link ConfiguredTargetFunction}'s logic for determining each target's
 * {@link BuildConfiguration}.
 *
 * <p>This is essentially an integration test for
 * {@link ConfiguredTargetFunction#computeDependencies} and {@link DependencyResolver}. These
 * methods form the core logic that figures out what a target's deps are, how their configurations
 * should differ from their parent, and how to instantiate those configurations as tangible
 * {@link BuildConfiguration} objects.
 *
 * <p>{@link ConfiguredTargetFunction} is a complicated class that does a lot of things. This test
 * focuses purely on the task of determining configurations for deps. So instead of evaluating
 * full {@link ConfiguredTargetFunction} instances, it evaluates a mock {@link SkyFunction} that
 * just wraps the {@link ConfiguredTargetFunction#computeDependencies} part. This keeps focus tight
 * and integration dependencies narrow.
 *
 * <p>We can't just call {@link ConfiguredTargetFunction#computeDependencies} directly because that
 * method needs a {@link SkyFunction.Environment} and Blaze's test infrastructure doesn't support
 * direct access to environments.
 */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsTest extends AnalysisTestCase {

  /**
   * A mock {@link SkyFunction} that just calls {@link ConfiguredTargetFunction#computeDependencies}
   * and returns its results.
   */
  private static class ComputeDependenciesFunction implements SkyFunction {
    static final SkyFunctionName SKYFUNCTION_NAME =
        SkyFunctionName.createHermetic("CONFIGURED_TARGET_FUNCTION_COMPUTE_DEPENDENCIES");

    private final LateBoundStateProvider stateProvider;
    private final Supplier<BuildOptions> buildOptionsSupplier;

    ComputeDependenciesFunction(
        LateBoundStateProvider lateBoundStateProvider,
        Supplier<BuildOptions> buildOptionsSupplier) {
      this.stateProvider = lateBoundStateProvider;
      this.buildOptionsSupplier = buildOptionsSupplier;
    }

    /** Returns a {@link SkyKey} for a given <Target, BuildConfiguration> pair. */
    private static Key key(Target target, BuildConfiguration config) {
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

    /**
     * Returns a {@link OrderedSetMultimap<Attribute, ConfiguredTarget>} map representing the
     * deps of given target.
     */
    static class Value implements SkyValue {
      OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap;

      Value(OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap) {
        this.depMap = depMap;
      }
    }

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws EvalException, InterruptedException {
      try {
        OrderedSetMultimap<DependencyKind, ConfiguredTargetAndData> depMap =
            ConfiguredTargetFunction.computeDependencies(
                env,
                new SkyframeDependencyResolver(env),
                (TargetAndConfiguration) skyKey.argument(),
                ImmutableList.<Aspect>of(),
                ImmutableMap.<Label, ConfigMatchingProvider>of(),
                /* toolchainContexts= */ null,
                /* useToolchainTransition= */ false,
                stateProvider.lateBoundRuleClassProvider(),
                stateProvider.lateBoundHostConfig(),
                NestedSetBuilder.<Package>stableOrder(),
                NestedSetBuilder.<Cause>stableOrder(),
                buildOptionsSupplier.get());
        return env.valuesMissing() ? null : new Value(depMap);
      } catch (RuntimeException e) {
        throw e;
      } catch (Exception e) {
        throw new EvalException(e);
      }
    }

    private static class EvalException extends SkyFunctionException {
      public EvalException(Exception cause) {
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
  private class LateBoundStateProvider {
    RuleClassProvider lateBoundRuleClassProvider() {
      return ruleClassProvider;
    }
    BuildConfiguration lateBoundHostConfig() {
      return getHostConfiguration();
    }
  }

  /**
   * An {@link AnalysisMock} that injects {@link ComputeDependenciesFunction} into the Skyframe
   * executor.
   */
  private static final class AnalysisMockWithComputeDepsFunction extends AnalysisMock.Delegate {
    private final LateBoundStateProvider stateProvider;
    private final Supplier<BuildOptions> defaultBuildOptions;

    AnalysisMockWithComputeDepsFunction(
        LateBoundStateProvider stateProvider, Supplier<BuildOptions> defaultBuildOptions) {
      super(AnalysisMock.get());
      this.stateProvider = stateProvider;
      this.defaultBuildOptions = defaultBuildOptions;
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions(
        BlazeDirectories directories) {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions(directories))
          .put(
              ComputeDependenciesFunction.SKYFUNCTION_NAME,
              new ComputeDependenciesFunction(stateProvider, defaultBuildOptions))
          .build();
    }
  };

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithComputeDepsFunction(
        new LateBoundStateProvider(), () -> skyframeExecutor.getDefaultBuildOptions());
  }

  /** Returns the configured deps for a given target. */
  private Multimap<DependencyKind, ConfiguredTargetAndData> getConfiguredDeps(
      ConfiguredTarget target) throws Exception {
    String targetLabel = AliasProvider.getDependencyLabel(target).toString();
    SkyKey key = ComputeDependenciesFunction.key(getTarget(targetLabel), getConfiguration(target));
    // Must re-enable analysis for Skyframe functions that create configured targets.
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
    Object evalResult = SkyframeExecutorTestUtils.evaluate(
        skyframeExecutor, key, /*keepGoing=*/false, reporter);
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    @SuppressWarnings("unchecked")
    SkyValue value = ((EvaluationResult<ComputeDependenciesFunction.Value>) evalResult).get(key);
    return ((ComputeDependenciesFunction.Value) value).depMap;
  }

  /**
   * Returns the configured deps for a given target under the given attribute. Assumes the target
   * uses the target configuration.
   *
   * <p>Throws an exception if the attribute can't be found.
   */
  protected List<ConfiguredTarget> getConfiguredDeps(String targetLabel, String attrName)
      throws Exception {
    ConfiguredTarget target = Iterables.getOnlyElement(update(targetLabel).getTargetsToBuild());
    return getConfiguredDeps(target, attrName);
  }

  /**
   * Returns the configured deps for a given configured target under the given attribute.
   *
   * <p>Throws an exception if the attribute can't be found.
   */
  protected List<ConfiguredTarget> getConfiguredDeps(ConfiguredTarget target, String attrName)
      throws Exception {
    String targetLabel = AliasProvider.getDependencyLabel(target).toString();
    Multimap<DependencyKind, ConfiguredTargetAndData> allDeps = getConfiguredDeps(target);
    for (DependencyKind kind : allDeps.keySet()) {
      Attribute attribute = kind.getAttribute();
      if (attribute.getName().equals(attrName)) {
        return ImmutableList.copyOf(
            Collections2.transform(
                allDeps.get(kind), ConfiguredTargetAndData::getConfiguredTarget));
      }
    }
    throw new AssertionError(
        String.format("Couldn't find attribute %s for label %s", attrName, targetLabel));
  }

  @Test
  public void nullConfiguredDepsHaveExpectedConfigs() throws Exception {
    scratch.file(
        "a/BUILD",
        "genrule(name = 'gen', srcs = ['gen.in'], cmd = '', outs = ['gen.out'])");
    ConfiguredTarget genIn = Iterables.getOnlyElement(getConfiguredDeps("//a:gen", "srcs"));
    assertThat(getConfiguration(genIn)).isNull();
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
    BuildConfiguration topLevelConfiguration =
        getConfiguration(Iterables.getOnlyElement(update("//a:binary").getTargetsToBuild()));
    for (ConfiguredTarget dep : deps) {
      assertThat(topLevelConfiguration.equalsOrIsSupersetOf(getConfiguration(dep))).isTrue();
    }
  }

  @Test
  public void hostDeps() throws Exception {
    scratch.file(
        "a/BUILD",
        "cc_binary(name = 'host_tool', srcs = ['host_tool.cc'])",
        "genrule(name = 'gen', srcs = [], cmd = '', outs = ['gen.out'], tools = [':host_tool'])");
    ConfiguredTarget toolDep = Iterables.getOnlyElement(getConfiguredDeps("//a:gen", "tools"));
    assertThat(getConfiguration(toolDep).isHostConfiguration()).isTrue();
  }

  @Test
  public void splitDeps() throws Exception {
    // This test does not pass with trimming because android_binary applies an aspect and aspects
    // are not yet correctly supported with trimming.
    if (defaultFlags().contains(Flag.TRIMMED_CONFIGURATIONS)) {
      return;
    }
    getAnalysisMock().ccSupport().setupCcToolchainConfigForCpu(mockToolsConfig, "armeabi-v7a");
    scratch.file(
        "java/a/BUILD",
        "cc_library(name = 'lib', srcs = ['lib.cc'])",
        "android_binary(name='a', manifest = 'AndroidManifest.xml', deps = [':lib'])");
    useConfiguration("--fat_apk_cpu=k8,armeabi-v7a");
    List<ConfiguredTarget> deps = getConfiguredDeps("//java/a:a", "deps");
    assertThat(deps).hasSize(2);
    ConfiguredTarget dep1 = deps.get(0);
    ConfiguredTarget dep2 = deps.get(1);
    assertThat(
            ImmutableList.<String>of(
                getConfiguration(dep1).getCpu(), getConfiguration(dep2).getCpu()))
        .containsExactly("armeabi-v7a", "k8");
    // We don't care what order split deps are listed, but it must be deterministic.
    assertThat(
            ConfigurationResolver.SPLIT_DEP_ORDERING.compare(
                Dependency.builder()
                    .setLabel(dep1.getLabel())
                    .setConfiguration(getConfiguration(dep1))
                    .build(),
                Dependency.builder()
                    .setLabel(dep2.getLabel())
                    .setConfiguration(getConfiguration(dep2))
                    .build()))
        .isLessThan(0);
  }

  /**
   * {@link ConfigurationResolver#resolveConfigurations} caches the transitions applied to deps. In
   * other words, if a parent rule has 100 deps that all set { compilation_mode=dbg }, there's no
   * need to compute that transition and request the resulting dep configuration from Skyframe 100
   * times.
   *
   * <p>But we do need to make sure <bold>different</bold> transitions don't trigger false cache
   * hits. This test checks a subtle version of that: if the same Starlark transition applies to two
   * deps, but that transition reads their attributes and their attribute values are different, we
   * need to make sure they're distinctly computed.
   */
  @Test
  public void sameTransitionDifferentParameters() throws Exception {
    scratch.file(
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
