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
import static org.junit.Assert.fail;

import com.google.common.base.VerifyException;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimap;
import com.google.common.collect.SetMultimap;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.DependencyResolver;
import com.google.devtools.build.lib.analysis.TargetAndConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.ConfigMatchingProvider;
import com.google.devtools.build.lib.analysis.util.AnalysisMock;
import com.google.devtools.build.lib.analysis.util.AnalysisTestCase;
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
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyFunction;
import com.google.devtools.build.skyframe.SkyFunctionException;
import com.google.devtools.build.skyframe.SkyFunctionName;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Collection;
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
         SkyFunctionName.create("CONFIGURED_TARGET_FUNCTION_COMPUTE_DEPENDENCIES");

    private final LateBoundStateProvider stateProvider;

    ComputeDependenciesFunction(LateBoundStateProvider lateBoundStateProvider) {
      this.stateProvider = lateBoundStateProvider;
    }

    /**
     * Returns a {@link SkyKey} for a given <Target, BuildConfiguration> pair.
     */
    static SkyKey key(Target target, BuildConfiguration config) {
      return SkyKey.create(SKYFUNCTION_NAME, new TargetAndConfiguration(target, config));
    }

    /**
     * Returns a {@link OrderedSetMultimap<Attribute, ConfiguredTarget>} map representing the
     * deps of given target.
     */
    static class Value implements SkyValue {
      OrderedSetMultimap<Attribute, ConfiguredTarget> depMap;
      Value(OrderedSetMultimap<Attribute, ConfiguredTarget> depMap) {
        this.depMap = depMap;
      }
    }

    @Override
    public SkyValue compute(SkyKey skyKey, Environment env)
        throws EvalException, InterruptedException {
      try {
        OrderedSetMultimap<Attribute, ConfiguredTarget> depMap =
            ConfiguredTargetFunction.computeDependencies(
                env,
                new SkyframeDependencyResolver(env),
                (TargetAndConfiguration) skyKey.argument(),
                ImmutableList.<Aspect>of(),
                ImmutableMap.<Label, ConfigMatchingProvider>of(),
                stateProvider.lateBoundRuleClassProvider(),
                stateProvider.lateBoundHostConfig(),
                NestedSetBuilder.<Package>stableOrder(),
                NestedSetBuilder.<Label>stableOrder());
        return env.valuesMissing() ? null : new Value(depMap);
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
    AnalysisMockWithComputeDepsFunction(LateBoundStateProvider stateProvider) {
      super(AnalysisMock.get());
      this.stateProvider = stateProvider;
    }

    @Override
    public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions() {
      return ImmutableMap.<SkyFunctionName, SkyFunction>builder()
          .putAll(super.getSkyFunctions())
          .put(
              ComputeDependenciesFunction.SKYFUNCTION_NAME,
              new ComputeDependenciesFunction(stateProvider))
          .build();
    }
  };

  @Override
  protected AnalysisMock getAnalysisMock() {
    return new AnalysisMockWithComputeDepsFunction(new LateBoundStateProvider());
  }

  /**
   * Returns the configured deps for a given target, assuming the target uses the target
   * configuration.
   */
  private Multimap<Attribute, ConfiguredTarget> getConfiguredDeps(String targetLabel)
      throws Exception {
    update(targetLabel);
    SkyKey key = ComputeDependenciesFunction.key(getTarget(targetLabel), getTargetConfiguration());
    // Must re-enable analysis for Skyframe functions that create configured targets.
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(true);
    Object evalResult = SkyframeExecutorTestUtils.evaluate(
        skyframeExecutor, key, /*keepGoing=*/false, reporter);
    skyframeExecutor.getSkyframeBuildView().enableAnalysis(false);
    SkyValue value = ((EvaluationResult<ComputeDependenciesFunction.Value>) evalResult).get(key);
    return ((ComputeDependenciesFunction.Value) value).depMap;
  }

  /**
   * Returns the configured deps for a given target under the given attribute. Assumes the target
   * uses the target configuration.
   *
   * <p>Throws an exception if the attribute can't be found.
   */
  private Collection<ConfiguredTarget> getConfiguredDeps(String targetLabel, String attrName)
      throws Exception {
    Multimap<Attribute, ConfiguredTarget> allDeps = getConfiguredDeps(targetLabel);
    for (Attribute attribute : allDeps.keySet()) {
      if (attribute.getName().equals(attrName)) {
        return allDeps.get(attribute);
      }
    }
    throw new AssertionError(
        String.format("Couldn't find attribute %s for label %s", attrName, targetLabel));
  }

  @Test
  public void putOnlyEntryCorrectWithSetMultimap() throws Exception {
    internalTestPutOnlyEntry(HashMultimap.<String, String>create());
  }

  /**
   * Unlike {@link SetMultimap}, {@link ListMultimap} allows duplicate <Key, value> pairs. Make
   * sure that doesn't fool {@link ConfiguredTargetFunction#putOnlyEntry}.
   */
  @Test
  public void putOnlyEntryCorrectWithListMultimap() throws Exception {
    internalTestPutOnlyEntry(ArrayListMultimap.<String, String>create());
  }

  private void internalTestPutOnlyEntry(Multimap<String, String> map) throws Exception {
    ConfiguredTargetFunction.putOnlyEntry(map, "foo", "bar");
    ConfiguredTargetFunction.putOnlyEntry(map, "baz", "bar");
    try {
      ConfiguredTargetFunction.putOnlyEntry(map, "foo", "baz");
      fail("Expected an exception when trying to add a new value to an existing key");
    } catch (VerifyException e) {
      assertThat(e).hasMessage("couldn't insert baz: map already has key foo");
    }
    try {
      ConfiguredTargetFunction.putOnlyEntry(map, "foo", "bar");
      fail("Expected an exception when trying to add a pre-existing <key, value> pair");
    } catch (VerifyException e) {
      assertThat(e).hasMessage("couldn't insert bar: map already has key foo");
    }
  }

  @Test
  public void nullConfiguredDepsHaveExpectedConfigs() throws Exception {
    scratch.file(
        "a/BUILD",
        "genrule(name = 'gen', srcs = ['gen.in'], cmd = '', outs = ['gen.out'])");
    ConfiguredTarget genIn = Iterables.getOnlyElement(getConfiguredDeps("//a:gen", "srcs"));
    assertThat(genIn.getConfiguration()).isNull();
  }

  /** Runs the same test with trimmed dynamic configurations. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithDynamicConfigurations extends ConfigurationsForTargetsTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.DYNAMIC_CONFIGURATIONS);
    }
  }

  /** Runs the same test with untrimmed dynamic configurations. */
  @TestSpec(size = Suite.SMALL_TESTS)
  @RunWith(JUnit4.class)
  public static class WithDynamicConfigurationsNoTrim extends ConfigurationsForTargetsTest {
    @Override
    protected FlagBuilder defaultFlags() {
      return super.defaultFlags().with(Flag.DYNAMIC_CONFIGURATIONS_NOTRIM);
    }
  }
}
