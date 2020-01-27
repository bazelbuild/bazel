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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ConfigurationResolver;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.ComposingTransition;
import com.google.devtools.build.lib.analysis.config.transitions.ConfigurationTransition;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.DummyRuleFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.List;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Runs an expanded set of ConfigurationsForTargetsTest with trimmed configurations. */
@TestSpec(size = Suite.SMALL_TESTS)
@RunWith(JUnit4.class)
public class ConfigurationsForTargetsWithTrimmedConfigurationsTest
    extends ConfigurationsForTargetsTest {
  @Override
  protected FlagBuilder defaultFlags() {
    return super.defaultFlags().with(Flag.TRIMMED_CONFIGURATIONS);
  }

  private static class NoopSplitTransition implements SplitTransition {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      return ImmutableList.of(buildOptions);
    }
  }

  private static class SetsHostCpuSplitTransition implements SplitTransition {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(CoreOptions.class).hostCpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }
  }

  private static class SetsCpuSplitTransition implements SplitTransition {

    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(CoreOptions.class).cpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }
  }

  private static class SetsCpuPatchTransition implements PatchTransition {

    @Override
    public BuildOptions patch(BuildOptions options) {
      BuildOptions result = options.clone();
      result.get(CoreOptions.class).cpu = "SET BY PATCH";
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
  private static final MockRule EMPTY_SPLIT_RULE =
      () ->
          MockRule.ancestor(TEST_BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "empty_split",
                  attr("with_empty_transition", LABEL)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .cfg(TransitionFactories.of(new NoopSplitTransition())));

  /** Rule with a split transition on an attribute. */
  private static final MockRule ATTRIBUTE_TRANSITION_RULE =
      () ->
          MockRule.ancestor(TEST_BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "attribute_transition",
                  attr("without_transition", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
                  attr("with_cpu_transition", LABEL)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .cfg(TransitionFactories.of(new SetsCpuSplitTransition())),
                  attr("with_host_cpu_transition", LABEL)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .cfg(TransitionFactories.of(new SetsHostCpuSplitTransition())));

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
    public BuildOptions patch(BuildOptions options) {
      BuildOptions result = options.clone();
      result.get(TestConfiguration.TestOptions.class).testFilter = "SET BY PATCH FACTORY: " + value;
      return result;
    }
  }

  @AutoCodec.VisibleForSerialization
  @AutoCodec
  static class SetsTestFilterFromAttributeTransitionFactory implements TransitionFactory<Rule> {
    @Override
    public PatchTransition create(Rule rule) {
      NonconfigurableAttributeMapper attributes = NonconfigurableAttributeMapper.of(rule);
      String value = attributes.get("sets_test_filter_to", STRING);
      if (Strings.isNullOrEmpty(value)) {
        return NoTransition.INSTANCE;
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

  private static final class AddArgumentToTestArgsTransition implements PatchTransition {
    private final String argument;

    public AddArgumentToTestArgsTransition(String argument) {
      this.argument = argument;
    }

    @Override
    public BuildOptions patch(BuildOptions options) {
      if (!options.contains(TestConfiguration.TestOptions.class)) {
        return options;
      }

      BuildOptions result = options.clone();
      TestConfiguration.TestOptions testOpts = result.get(TestConfiguration.TestOptions.class);
      testOpts.testArguments =
          new ImmutableList.Builder<String>().addAll(testOpts.testArguments).add(argument).build();
      return result;
    }
  }

  /** Rule which adds an argument to the --test_args flag for its dependencies. */
  private static final MockRule ADD_TEST_ARG_FOR_DEPS_RULE =
      () ->
          MockRule.ancestor(TEST_BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "add_test_arg_for_deps",
                  attr("deps", LABEL_LIST)
                      .allowedFileTypes(FileTypeSet.ANY_FILE)
                      .cfg(
                          TransitionFactories.of(
                              new AddArgumentToTestArgsTransition("deps transition"))));

  /** Rule which adds an argument to the --test_args flag for itself. */
  private static final MockRule ADD_TEST_ARG_FOR_SELF_RULE =
      () ->
          MockRule.ancestor(TEST_BASE_RULE.getClass())
              .factory(DummyRuleFactory.class)
              .define(
                  "add_test_arg_for_self",
                  (builder, env) ->
                      builder
                          .cfg(new AddArgumentToTestArgsTransition("rule class transition"))
                          .add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));

  @Test
  public void trimmingTransitionActivatesLastOnAllTargets() throws Exception {
    TransitionFactory<Rule> trimmingTransitionFactory =
        (rule) ->
            new AddArgumentToTestArgsTransition(
                "trimming transition for " + rule.getLabel().toString());
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(TestAspects.BASE_RULE);
    builder.addRuleDefinition(TEST_BASE_RULE);
    builder.addRuleDefinition(ADD_TEST_ARG_FOR_DEPS_RULE);
    builder.addRuleDefinition(ADD_TEST_ARG_FOR_SELF_RULE);
    builder.overrideTrimmingTransitionFactoryForTesting(trimmingTransitionFactory);
    useRuleClassProvider(builder.build());
    scratch.file(
        "a/skylark.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    '_base': attr.label(default = '//a:base'),",
        "  }",
        ")");
    scratch.file(
        "a/BUILD",
        "load(':skylark.bzl', 'skylark_rule')",
        // ensure that all Skylark rules get the TestConfiguration fragment
        "test_base(name = 'base')",
        // skylark rules get trimmed
        "skylark_rule(name = 'skylark_solo', deps = [':base'])",
        // native rules get trimmed; top-level targets get trimmed after the rule-class transition
        "add_test_arg_for_self(name = 'test_arg_on_self')",
        // deps with dependency transitions get trimmed after the dependency transition
        "add_test_arg_for_deps(name = 'attribute_transition', deps = [':dep_after_transition'])",
        "skylark_rule(name = 'dep_after_transition')",
        // deps on rule-class transitions get trimmed after the rule-class transition
        "skylark_rule(name = 'dep_on_ruleclass', deps = [':ruleclass_transition'])",
        "add_test_arg_for_self(name = 'ruleclass_transition')",
        // when all three (rule-class, attribute, trimming transitions) collide it's okay
        "add_test_arg_for_deps(name = 'attribute_outer', deps = [':ruleclass_inner'])",
        "add_test_arg_for_self(name = 'ruleclass_inner')");

    ConfiguredTarget configuredTarget;
    BuildConfiguration config;

    configuredTarget = Iterables.getOnlyElement(update("//a:skylark_solo").getTargetsToBuild());
    config = getConfiguration(configuredTarget);
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly("trimming transition for //a:skylark_solo");

    configuredTarget = Iterables.getOnlyElement(update("//a:test_arg_on_self").getTargetsToBuild());
    config = getConfiguration(configuredTarget);
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly("rule class transition", "trimming transition for //a:test_arg_on_self")
        .inOrder();

    configuredTarget =
        Iterables.getOnlyElement(update("//a:attribute_transition").getTargetsToBuild());
    config =
        getConfiguration(
            Iterables.getOnlyElement(getConfiguredDeps(configuredTarget, "deps")));
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly(
            "trimming transition for //a:attribute_transition",
            "deps transition",
            "trimming transition for //a:dep_after_transition")
        .inOrder();

    configuredTarget = Iterables.getOnlyElement(update("//a:dep_on_ruleclass").getTargetsToBuild());
    config =
        getConfiguration(
            Iterables.getOnlyElement(getConfiguredDeps(configuredTarget, "deps")));
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly(
            "trimming transition for //a:dep_on_ruleclass",
            "rule class transition",
            "trimming transition for //a:ruleclass_transition")
        .inOrder();

    configuredTarget = Iterables.getOnlyElement(update("//a:attribute_outer").getTargetsToBuild());
    config =
        getConfiguration(
            Iterables.getOnlyElement(getConfiguredDeps(configuredTarget, "deps")));
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly(
            "trimming transition for //a:attribute_outer",
            "deps transition",
            "rule class transition",
            "trimming transition for //a:ruleclass_inner")
        .inOrder();
  }

  @Test
  public void trimmingTransitionsAreComposedInOrderOfAdding() throws Exception {
    TransitionFactory<Rule> firstTrimmingTransitionFactory =
        (rule) ->
            new AddArgumentToTestArgsTransition(
                "first trimming transition for " + rule.getLabel().toString());
    TransitionFactory<Rule> secondTrimmingTransitionFactory =
        (rule) ->
            new AddArgumentToTestArgsTransition(
                "second trimming transition for " + rule.getLabel().toString());
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(TestAspects.BASE_RULE);
    builder.addRuleDefinition(TEST_BASE_RULE);
    builder.overrideTrimmingTransitionFactoryForTesting(firstTrimmingTransitionFactory);
    builder.addTrimmingTransitionFactory(secondTrimmingTransitionFactory);
    useRuleClassProvider(builder.build());
    scratch.file(
        "a/skylark.bzl",
        "def _impl(ctx):",
        "  return",
        "skylark_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'deps': attr.label_list(),",
        "    '_base': attr.label(default = '//a:base'),",
        "  }",
        ")");
    scratch.file(
        "a/BUILD",
        "load(':skylark.bzl', 'skylark_rule')",
        // ensure that all Skylark rules get the TestConfiguration fragment
        "test_base(name = 'base')",
        // skylark rules get trimmed
        "skylark_rule(name = 'skylark_solo', deps = [':base'])");

    ConfiguredTarget configuredTarget;
    BuildConfiguration config;

    configuredTarget = Iterables.getOnlyElement(update("//a:skylark_solo").getTargetsToBuild());
    config = getConfiguration(configuredTarget);
    assertThat(config.getFragment(TestConfiguration.class).getTestArguments())
        .containsExactly(
            "first trimming transition for //a:skylark_solo",
            "second trimming transition for //a:skylark_solo")
        .inOrder();
  }

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
    BuildConfiguration ruleclass = getConfiguration(Iterables.getOnlyElement(deps));
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
    BuildConfiguration ruleclass = getConfiguration(Iterables.getOnlyElement(deps));
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
    BuildConfiguration ruleclass = getConfiguration(Iterables.getOnlyElement(deps));
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
    BuildConfiguration ruleclass = getConfiguration(Iterables.getOnlyElement(deps));
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
    assertThat(getConfiguration(target).getCpu()).isEqualTo("SET BY PATCH");
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
    assertThat(getConfiguration(target).getCpu()).isNotEqualTo("SET BY PATCH");
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
    BuildConfiguration config = getConfiguration(Iterables.getOnlyElement(deps));
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
    BuildConfiguration config = getConfiguration(Iterables.getOnlyElement(deps));
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
    assertThat(getConfiguration(target).getFragment(TestConfiguration.class).getTestFilter())
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
    assertThat(getConfiguration(target).getFragment(TestConfiguration.class).getTestFilter())
        .isEqualTo("SET ON COMMAND LINE: original and best");
  }

  /**
   * Returns a custom {@link PatchTransition} with the given value added to {@link
   * TestConfiguration.TestOptions#testFilter}.
   */
  private static PatchTransition newPatchTransition(final String value) {
    return new PatchTransition() {
      @Override
      public BuildOptions patch(BuildOptions options) {
        BuildOptions toOptions = options.clone();
        TestConfiguration.TestOptions baseOptions =
            toOptions.get(TestConfiguration.TestOptions.class);
        baseOptions.testFilter = nullToEmpty(baseOptions.testFilter) + value;
        return toOptions;
      }
    };
  }

  /**
   * Returns a custom {@link SplitTransition} that splits {@link
   * TestConfiguration.TestOptions#testFilter} down two paths: {@code += prefix + "1"} and {@code +=
   * prefix + "2"}.
   */
  private static SplitTransition newSplitTransition(final String prefix) {
    return buildOptions -> {
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
    };
  }

  /**
   * Returns the value of {@link TestConfiguration.TestOptions#testFilter} for a transition
   * applied over the target configuration.
   */
  private List<String> getTestFilterOptionValue(ConfigurationTransition transition)
      throws Exception {
    ImmutableList.Builder<String> outValues = ImmutableList.builder();
    for (BuildOptions toOptions :
        ConfigurationResolver.applyTransition(
            getTargetConfiguration().getOptions(),
            transition,
            ImmutableMap.of(),
            NullEventHandler.INSTANCE)) {
      outValues.add(toOptions.get(TestConfiguration.TestOptions.class).testFilter);
    }
    return outValues.build();
  }

  @Test
  public void composedStraightTransitions() throws Exception {
    update(); // Creates the target configuration.
    assertThat(
            getTestFilterOptionValue(
                ComposingTransition.of(newPatchTransition("foo"), newPatchTransition("bar"))))
        .containsExactly("foobar");
  }

  @Test
  public void composedStraightTransitionThenSplitTransition() throws Exception {
    update(); // Creates the target configuration.
    assertThat(
            getTestFilterOptionValue(
                ComposingTransition.of(newPatchTransition("foo"), newSplitTransition("split"))))
        .containsExactly("foosplit1", "foosplit2");
  }

  @Test
  public void composedSplitTransitionThenStraightTransition() throws Exception {
    update(); // Creates the target configuration.
    assertThat(
            getTestFilterOptionValue(
                ComposingTransition.of(newSplitTransition("split"), newPatchTransition("foo"))))
        .containsExactly("split1foo", "split2foo");
  }

  @Test
  public void composedSplitTransitions() throws Exception {
    update(); // Creates the target configuration.
    assertThat(
            getTestFilterOptionValue(
                ComposingTransition.of(newSplitTransition("s"), newSplitTransition("t"))))
        .containsExactly("s1t1", "s1t2", "s2t1", "s2t2");
  }

}
