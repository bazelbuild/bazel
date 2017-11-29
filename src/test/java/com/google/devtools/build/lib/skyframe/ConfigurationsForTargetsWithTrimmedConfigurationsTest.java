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
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.STRING;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.config.TransitionResolver;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.analysis.util.MockRuleDefaults;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.DummyRuleFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Runs an expanded set of ConfigurationsForTargetsTest with trimmed configurations. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsWithTrimmedConfigurationsTest
    extends ConfigurationsForTargetsTest {

  private TransitionResolver transitionResolver;

  @Before
  public void createTransitionResolver() {
    transitionResolver = new TransitionResolver(ruleClassProvider.getDynamicTransitionMapper());
  }

  @Override
  protected FlagBuilder defaultFlags() {
    return super.defaultFlags().with(Flag.TRIMMED_CONFIGURATIONS);
  }

  private static class EmptySplitTransition implements SplitTransition<BuildOptions> {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      return ImmutableList.of();
    }
  }

  private static class SetsHostCpuSplitTransition implements SplitTransition<BuildOptions> {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(BuildConfiguration.Options.class).hostCpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }
  }

  private static class SetsCpuSplitTransition implements SplitTransition<BuildOptions> {

    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(BuildConfiguration.Options.class).cpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }
  }

  private static class SetsCpuPatchTransition implements PatchTransition {

    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions result = options.clone();
      result.get(BuildConfiguration.Options.class).cpu = "SET BY PATCH";
      return result;
    }
  }

  /** Base rule that depends on the test configuration fragment. */
  private static final MockRule TEST_BASE_RULE = () ->
      MockRule.ancestor(TestAspects.BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "test_base",
          (builder, env) ->
              builder.requiresConfigurationFragments(TestConfiguration.class).build());

  /** A rule with an empty split transition on an attribute. */
  private static final MockRule EMPTY_SPLIT_RULE = () ->
      MockRule.ancestor(TEST_BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "empty_split",
          attr("with_empty_transition", LABEL)
              .allowedFileTypes(FileTypeSet.ANY_FILE)
              .cfg(new EmptySplitTransition()));

  /** Rule with a split transition on an attribute. */
  private static final MockRule ATTRIBUTE_TRANSITION_RULE = () ->
      MockRule.ancestor(TEST_BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "attribute_transition",
          attr("without_transition", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
          attr("with_cpu_transition", LABEL)
              .allowedFileTypes(FileTypeSet.ANY_FILE)
              .cfg(new SetsCpuSplitTransition()),
          attr("with_host_cpu_transition", LABEL)
              .allowedFileTypes(FileTypeSet.ANY_FILE)
              .cfg(new SetsHostCpuSplitTransition()));

  /** Rule with rule class configuration transition. */
  private static final MockRule RULE_CLASS_TRANSITION_RULE = () ->
      MockRule.ancestor(TEST_BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "rule_class_transition",
          (builder, env) -> builder.cfg(new SetsCpuPatchTransition()).build());

  private static class SetsTestFilterFromAttributePatchTransition implements PatchTransition {
    private final String value;

    public SetsTestFilterFromAttributePatchTransition(String value) {
      this.value = value;
    }

    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions result = options.clone();
      result.get(TestConfiguration.TestOptions.class).testFilter = "SET BY PATCH FACTORY: " + value;
      return result;
    }
  }

  private static class SetsTestFilterFromAttributeTransitionFactory
      implements RuleTransitionFactory {
    @Override
    public Transition buildTransitionFor(Rule rule) {
      NonconfigurableAttributeMapper attributes = NonconfigurableAttributeMapper.of(rule);
      String value = attributes.get("sets_test_filter_to", STRING);
      if (Strings.isNullOrEmpty(value)) {
        return null;
      } else {
        return new SetsTestFilterFromAttributePatchTransition(value);
      }
    }
  }

  /**
   * Rule with a RuleTransitionFactory which sets the --test_filter flag according to its attribute.
   */
  private static final MockRule USES_RULE_TRANSITION_FACTORY_RULE = () ->
      MockRule.ancestor(TEST_BASE_RULE.getClass()).factory(DummyRuleFactory.class).define(
          "uses_rule_transition_factory",
          (builder, env) ->
              builder
                  .cfg(new SetsTestFilterFromAttributeTransitionFactory())
                  .add(attr("sets_test_filter_to", STRING)
                      .nonconfigurable("used in RuleTransitionFactory")
                      .value("")));

  @Test
  public void testRuleClassTransition() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, ATTRIBUTE_TRANSITION_RULE,
        RULE_CLASS_TRANSITION_RULE);
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
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, ATTRIBUTE_TRANSITION_RULE,
        RULE_CLASS_TRANSITION_RULE);
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
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, ATTRIBUTE_TRANSITION_RULE,
        RULE_CLASS_TRANSITION_RULE);
    scratch.file("a/BUILD",
        "attribute_transition(",
        "   name='attribute',",
        "   with_cpu_transition = ':rule_class',",
        ")",
        "rule_class_transition(name='rule_class')");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:attribute", "with_cpu_transition");
    BuildConfiguration ruleclass = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(ruleclass.getCpu()).isEqualTo("SET BY PATCH");
  }

  @Test
  public void testEmptySplitDoesNotSuppressRuleClassTransition() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, EMPTY_SPLIT_RULE,
        RULE_CLASS_TRANSITION_RULE);
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
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, RULE_CLASS_TRANSITION_RULE);
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
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, RULE_CLASS_TRANSITION_RULE,
        TestAspects.SIMPLE_RULE);
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

  @Test
  public void ruleTransitionFactoryUsesNonconfigurableAttributesToGenerateTransition()
      throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, ATTRIBUTE_TRANSITION_RULE,
        USES_RULE_TRANSITION_FACTORY_RULE);
    useConfiguration("--test_filter=SET ON COMMAND LINE: original and best");
    scratch.file(
        "a/BUILD",
        "attribute_transition(",
        "   name='top',",
        "   without_transition=':factory',",
        ")",
        "uses_rule_transition_factory(",
        "   name='factory',",
        "   sets_test_filter_to='funkiest',",
        ")");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:top", "without_transition");
    BuildConfiguration config = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(config.getFragment(TestConfiguration.class).getTestFilter())
        .isEqualTo("SET BY PATCH FACTORY: funkiest");
  }

  @Test
  public void ruleTransitionFactoryCanReturnNullToCauseNoTransition() throws Exception {
    setRulesAvailableInTests(TestAspects.BASE_RULE, TEST_BASE_RULE, ATTRIBUTE_TRANSITION_RULE,
        USES_RULE_TRANSITION_FACTORY_RULE);
    useConfiguration("--test_filter=SET ON COMMAND LINE: original and best");
    scratch.file(
        "a/BUILD",
        "attribute_transition(",
        "   name='top',",
        "   without_transition=':factory',",
        ")",
        "uses_rule_transition_factory(",
        "   name='factory',",
        "   sets_test_filter_to='',",
        ")");
    List<ConfiguredTarget> deps = getConfiguredDeps("//a:top", "without_transition");
    BuildConfiguration config = Iterables.getOnlyElement(deps).getConfiguration();
    assertThat(config.getFragment(TestConfiguration.class).getTestFilter())
        .isEqualTo("SET ON COMMAND LINE: original and best");
  }

  @Test
  public void topLevelRuleTransitionFactoryUsesNonconfigurableAttributes() throws Exception {
    setRulesAvailableInTests(
        TestAspects.BASE_RULE, TEST_BASE_RULE, USES_RULE_TRANSITION_FACTORY_RULE);
    useConfiguration("--test_filter=SET ON COMMAND LINE: original and best");
    scratch.file(
        "a/BUILD",
        "uses_rule_transition_factory(",
        "   name='factory',",
        "   sets_test_filter_to='Maximum Dance',",
        ")");
    ConfiguredTarget target = Iterables.getOnlyElement(update("//a:factory").getTargetsToBuild());
    assertThat(target.getConfiguration().getFragment(TestConfiguration.class).getTestFilter())
        .isEqualTo("SET BY PATCH FACTORY: Maximum Dance");
  }

  @Test
  public void topLevelRuleTransitionFactoryCanReturnNullInTesting() throws Exception {
    setRulesAvailableInTests(
        TestAspects.BASE_RULE, TEST_BASE_RULE, USES_RULE_TRANSITION_FACTORY_RULE);
    useConfiguration("--test_filter=SET ON COMMAND LINE: original and best");
    scratch.file(
        "a/BUILD",
        "uses_rule_transition_factory(",
        "   name='factory',",
        "   sets_test_filter_to='',",
        ")");
    update("@//a:factory");
    ConfiguredTarget target = getView().getConfiguredTargetForTesting(
        reporter,
        Label.parseAbsoluteUnchecked("@//a:factory"),
        getTargetConfiguration());
    assertThat(target.getConfiguration().getFragment(TestConfiguration.class).getTestFilter())
        .isEqualTo("SET ON COMMAND LINE: original and best");
  }

  /**
   * Returns a custom {@link PatchTransition} with the given value added to {@link
   * TestConfiguration.TestOptions#testFilter}.
   */
  private static PatchTransition newPatchTransition(final String value) {
    return new PatchTransition() {
      @Override
      public BuildOptions apply(BuildOptions options) {
        BuildOptions toOptions = options.clone();
        TestConfiguration.TestOptions baseOptions =
            toOptions.get(TestConfiguration.TestOptions.class);
        baseOptions.testFilter = (nullToEmpty(baseOptions.testFilter)) + value;
        return toOptions;
      }
    };
  }

  /**
   * Returns a custom {@link Attribute.SplitTransition} that splits {@link
   * TestConfiguration.TestOptions#testFilter} down two paths: {@code += prefix + "1"} and {@code +=
   * prefix + "2"}.
   */
  private static Attribute.SplitTransition<BuildOptions> newSplitTransition(final String prefix) {
    return new Attribute.SplitTransition<BuildOptions>() {
      @Override
      public List<BuildOptions> split(BuildOptions buildOptions) {
        ImmutableList.Builder<BuildOptions> result = ImmutableList.builder();
        for (int index = 1; index <= 2; index++) {
          BuildOptions toOptions = buildOptions.clone();
          TestConfiguration.TestOptions baseOptions =
              toOptions.get(TestConfiguration.TestOptions.class);
          baseOptions.testFilter =
              (baseOptions.testFilter == null ? "" : baseOptions.testFilter) + prefix + index;
          result.add(toOptions);
        }
        return result.build();
      }
    };
  }

  /**
   * Returns the value of {@link TestConfiguration.TestOptions#testFilter} for a transition
   * applied over the target configuration.
   */
  private List<String> getTestFilterOptionValue(Transition transition)
      throws Exception {
    ImmutableList.Builder<String> outValues = ImmutableList.builder();
    for (BuildOptions toOptions : ConfigurationResolver.applyTransition(
        getTargetConfiguration().getOptions(), transition,
        ruleClassProvider.getAllFragments(), ruleClassProvider, false)) {
      outValues.add(toOptions.get(TestConfiguration.TestOptions.class).testFilter);
    }
    return outValues.build();
  }

  @Test
  public void composedStraightTransitions() throws Exception {
    update(); // Creates the target configuration.
    assertThat(getTestFilterOptionValue(
        transitionResolver.composeTransitions(
            newPatchTransition("foo"),
            newPatchTransition("bar"))))
        .containsExactly("foobar");
  }

  @Test
  public void composedStraightTransitionThenSplitTransition() throws Exception {
    update(); // Creates the target configuration.
    assertThat(getTestFilterOptionValue(
        transitionResolver.composeTransitions(
            newPatchTransition("foo"),
            newSplitTransition("split"))))
        .containsExactly("foosplit1", "foosplit2");
  }

  @Test
  public void composedSplitTransitionThenStraightTransition() throws Exception {
    update(); // Creates the target configuration.
    assertThat(getTestFilterOptionValue(
        transitionResolver.composeTransitions(
            newSplitTransition("split"),
            newPatchTransition("foo"))))
        .containsExactly("split1foo", "split2foo");
  }

  @Test
  public void composedSplitTransitions() throws Exception {
    update(); // Creates the target configuration.
    assertThat(getTestFilterOptionValue(
        transitionResolver.composeTransitions(
            newSplitTransition("s"),
            newSplitTransition("t"))))
        .containsExactly("s1t1", "s1t2", "s2t1", "s2t2");
  }

  /** Sets {@link TestConfiguration.TestOptions#testFilter} to the rule class of the given rule. */
  private static final RuleTransitionFactory RULE_BASED_TEST_FILTER =
      rule ->
          (PatchTransition)
              buildOptions -> {
                BuildOptions toOptions = buildOptions.clone();
                toOptions.get(TestConfiguration.TestOptions.class).testFilter = rule.getRuleClass();
                return toOptions;
              };

  private static final RuleDefinition RULE_WITH_OUTGOING_TRANSITION =
      (MockRule)
          () ->
              MockRule.define(
                  "change_deps",
                  (builder, env) ->
                      builder
                          .add(MockRuleDefaults.DEPS_ATTRIBUTE)
                          .requiresConfigurationFragments(TestConfiguration.class)
                          .depsCfg(RULE_BASED_TEST_FILTER));

  @Test
  public void outgoingRuleTransition() throws Exception {
    setRulesAvailableInTests(
        RULE_WITH_OUTGOING_TRANSITION,
        (MockRule)
            () ->
                MockRule.define(
                    "foo_rule",
                    (builder, env) ->
                        builder.requiresConfigurationFragments(TestConfiguration.class)),
        (MockRule)
            () ->
                MockRule.define(
                    "bar_rule",
                    (builder, env) ->
                        builder.requiresConfigurationFragments(TestConfiguration.class)));
    scratch.file("outgoing/BUILD",
        "foo_rule(",
        "    name = 'foolib')",
        "bar_rule(",
        "    name = 'barlib')",
        "change_deps(",
        "    name = 'bin',",
        "    deps  = [':foolib', ':barlib'])");

    List<ConfiguredTarget> deps = getConfiguredDeps("//outgoing:bin", "deps");
    ImmutableMap<String, String> depLabelToTestFilterString =
        ImmutableMap.of(
            deps.get(0).getLabel().toString(),
                deps.get(0).getConfiguration().getFragment(TestConfiguration.class).getTestFilter(),
            deps.get(1).getLabel().toString(),
                deps.get(1)
                    .getConfiguration()
                    .getFragment(TestConfiguration.class)
                    .getTestFilter());

    assertThat(depLabelToTestFilterString).containsExactly(
        "//outgoing:foolib", "foo_rule",
        "//outgoing:barlib", "bar_rule");
  }
}
