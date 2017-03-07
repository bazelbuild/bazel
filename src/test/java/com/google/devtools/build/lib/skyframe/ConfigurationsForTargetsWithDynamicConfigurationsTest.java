// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Strings.nullToEmpty;
import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.fail;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import java.util.ArrayList;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Runs an expanded set of ConfigurationsForTargetsTest with trimmed dynamic configurations. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsWithDynamicConfigurationsTest
    extends ConfigurationsForTargetsTest {
  @Override
  protected FlagBuilder defaultFlags() {
    return super.defaultFlags().with(Flag.DYNAMIC_CONFIGURATIONS);
  }

  @Test
  public void testRuleClassTransition() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   without_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "without_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
  }

  @Test
  public void testNonConflictingAttributeAndRuleClassTransitions() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   with_host_cpu_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "with_host_cpu_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
    assertThat(ruleclass.getHostCpu()).isEqualTo("SET BY SPLIT");
  }

  @Test
  public void testConflictingAttributeAndRuleClassTransitions() throws Exception {
    setRulesAvailableInTests(new TestAspects.BaseRule(),
        new TestAspects.AttributeTransitionRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   with_cpu_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "with_cpu_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY SPLIT");
  }

  @Test
  public void testEmptySplitDoesNotSuppressRuleClassTransition() throws Exception {
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestAspects.EmptySplitRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file(
        "a/BUILD",
        "empty_split(",
        "   name = 'empty',",
        "   with_empty_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:empty", "with_empty_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
  }

  @Test
  public void testTopLevelRuleClassTransition() throws Exception {
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestAspects.RuleClassTransitionRule());
    scratch.file(
        "a/BUILD",
        "rule_class_transition(",
        "   name = 'rule_class',",
        ")");
    ConfiguredTarget target =
        Iterables.getOnlyElement(update("//a:rule_class").getTargetsToBuild());
    assertThat(target.getConfiguration().getCpu()).isEqualTo("SET BY PATCH");
  }

  @Test
  public void testTopLevelRuleClassTransitionAndNoTransition() throws Exception {
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestAspects.RuleClassTransitionRule(),
        new TestAspects.SimpleRule());
    scratch.file(
        "a/BUILD",
        "rule_class_transition(",
        "   name = 'rule_class',",
        ")",
        "simple(name='sim')");
    ConfiguredTarget target =
        Iterables.getOnlyElement(update("//a:sim").getTargetsToBuild());
    assertThat(target.getConfiguration().getCpu()).isNotEqualTo("SET BY PATCH");
  }

  /**
   * Returns a custom {@link PatchTransition} with the given value added to
   * {@link BuildConfiguration.Options#testFilter}.
   */
  private static PatchTransition newPatchTransition(final String value) {
    return new PatchTransition() {
      @Override
      public BuildOptions apply(BuildOptions options) {
        BuildOptions toOptions = options.clone();
        BuildConfiguration.Options baseOptions = toOptions.get(BuildConfiguration.Options.class);
        baseOptions.testFilter = (nullToEmpty(baseOptions.testFilter)) + value;
        return toOptions;
      }

      @Override
      public boolean defaultsToSelf() {
        return false;
      }
    };
  }

  /**
   * Returns a custom {@link Attribute.SplitTransition} that splits
   * {@link BuildConfiguration.Options#testFilter} down two paths: {@code += prefix + "1"}
   * and {@code += prefix + "2"}.
   */
  private static Attribute.SplitTransition<BuildOptions> newSplitTransition(final String prefix) {
    return new Attribute.SplitTransition<BuildOptions>() {
      @Override
      public List<BuildOptions> split(BuildOptions buildOptions) {
        ImmutableList.Builder<BuildOptions> result = ImmutableList.builder();
        for (int index = 1; index <= 2; index++) {
          BuildOptions toOptions = buildOptions.clone();
          BuildConfiguration.Options baseOptions = toOptions.get(BuildConfiguration.Options.class);
          baseOptions.testFilter =
              (baseOptions.testFilter == null ? "" : baseOptions.testFilter) + prefix + index;
          result.add(toOptions);
        }
        return result.build();
      }

      @Override
      public boolean defaultsToSelf() {
        return false;
      }
    };
  }

  @Test
  public void composedStraightTransitions() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyTransition(newPatchTransition("foo"));
    applier.applyTransition(newPatchTransition("bar"));
    Dependency dep = Iterables.getOnlyElement(
        applier.getDependencies(Label.create("some", "target"), AspectCollection.EMPTY));
    BuildOptions toOptions = Iterables.getOnlyElement(
        ConfiguredTargetFunction.getDynamicTransitionOptions(getTargetConfiguration().getOptions(),
            dep.getTransition(), ruleClassProvider.getAllFragments(), ruleClassProvider, false));
    assertThat(toOptions.get(BuildConfiguration.Options.class).testFilter).isEqualTo("foobar");
  }

  @Test
  public void composedStraightTransitionThenSplitTransition() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyTransition(newPatchTransition("foo"));
    applier.split(newSplitTransition("split"));
    Dependency dep = Iterables.getOnlyElement(
        applier.getDependencies(Label.create("some", "target"), AspectCollection.EMPTY));
    List<String> outValues = new ArrayList<>();
    for (BuildOptions toOptions : ConfiguredTargetFunction.getDynamicTransitionOptions(
        getTargetConfiguration().getOptions(), dep.getTransition(),
        ruleClassProvider.getAllFragments(), ruleClassProvider, false)) {
      outValues.add(toOptions.get(BuildConfiguration.Options.class).testFilter);
    }
    assertThat(outValues).containsExactly("foosplit1", "foosplit2");
  }

  @Test
  public void composedSplitTransitionThenStraightTransition() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.split(newSplitTransition("split"));
    applier.applyTransition(newPatchTransition("foo"));
    Dependency dep = Iterables.getOnlyElement(
        applier.getDependencies(Label.create("some", "target"), AspectCollection.EMPTY));
    List<String> outValues = new ArrayList<>();
    for (BuildOptions toOptions : ConfiguredTargetFunction.getDynamicTransitionOptions(
        getTargetConfiguration().getOptions(), dep.getTransition(),
        ruleClassProvider.getAllFragments(), ruleClassProvider, false)) {
      outValues.add(toOptions.get(BuildConfiguration.Options.class).testFilter);
    }
    assertThat(outValues).containsExactly("split1foo", "split2foo");
  }

  @Test
  public void composedSplitTransitions() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.split(newSplitTransition("split"));
    try {
      applier.split(newSplitTransition("disallowed second split"));
      fail("expected failure: deps cannot apply more than one split transition each");
    } catch (IllegalStateException e) {
      assertThat(e.getMessage()).contains("dependency edges may apply at most one split");
    }
  }
}
