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
import static com.google.devtools.build.lib.packages.Attribute.ANY_RULE;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.syntax.Type.STRING;
import static org.junit.Assert.fail;

import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.AspectCollection;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Dependency;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.PatchTransition;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.analysis.util.TestAspects;
import com.google.devtools.build.lib.analysis.util.TestAspects.DummyRuleFactory;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.SplitTransition;
import com.google.devtools.build.lib.packages.Attribute.Transition;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleTransitionFactory;
import com.google.devtools.build.lib.rules.test.TestConfiguration;
import com.google.devtools.build.lib.testutil.Suite;
import com.google.devtools.build.lib.testutil.TestSpec;
import com.google.devtools.build.lib.util.FileTypeSet;
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

  private static class EmptySplitTransition implements SplitTransition<BuildOptions> {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      return ImmutableList.of();
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }

  private static class SetsHostCpuSplitTransition implements SplitTransition<BuildOptions> {
    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(BuildConfiguration.Options.class).hostCpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }

  private static class SetsCpuSplitTransition implements SplitTransition<BuildOptions> {

    @Override
    public List<BuildOptions> split(BuildOptions buildOptions) {
      BuildOptions result = buildOptions.clone();
      result.get(BuildConfiguration.Options.class).cpu = "SET BY SPLIT";
      return ImmutableList.of(result);
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }

  private static class SetsCpuPatchTransition implements PatchTransition {

    @Override
    public BuildOptions apply(BuildOptions options) {
      BuildOptions result = options.clone();
      result.get(BuildConfiguration.Options.class).cpu = "SET BY PATCH";
      return result;
    }

    @Override
    public boolean defaultsToSelf() {
      return true;
    }
  }

  /** Base rule that depends on the test configuration fragment. */
  private static class TestBaseRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder.requiresConfigurationFragments(TestConfiguration.class).build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("test_base")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(TestAspects.BaseRule.class)
          .build();
    }
  }

  /** A rule with an empty split transition on an attribute. */
  private static class EmptySplitRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(
              attr("with_empty_transition", LABEL)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .cfg(new EmptySplitTransition()))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("empty_split")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(TestBaseRule.class)
          .build();
    }
  }

  /** Rule with a split transition on an attribute. */
  private static class AttributeTransitionRule implements RuleDefinition {

    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .add(attr("without_transition", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE))
          .add(
              attr("with_cpu_transition", LABEL)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .cfg(new SetsCpuSplitTransition()))
          .add(
              attr("with_host_cpu_transition", LABEL)
                  .allowedFileTypes(FileTypeSet.ANY_FILE)
                  .cfg(new SetsHostCpuSplitTransition()))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("attribute_transition")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(TestBaseRule.class)
          .build();
    }
  }

  /** Rule with rule class configuration transition. */
  private static class RuleClassTransitionRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder.cfg(new SetsCpuPatchTransition()).build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_class_transition")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(TestBaseRule.class)
          .build();
    }
  }

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

    @Override
    public boolean defaultsToSelf() {
      return true;
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
  private static class UsesRuleTransitionFactoryRule implements RuleDefinition {
    @Override
    public RuleClass build(RuleClass.Builder builder, RuleDefinitionEnvironment environment) {
      return builder
          .cfg(new SetsTestFilterFromAttributeTransitionFactory())
          .add(
              attr("sets_test_filter_to", STRING)
                  .nonconfigurable("used in RuleTransitionFactory")
                  .value(""))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("uses_rule_transition_factory")
          .factoryClass(DummyRuleFactory.class)
          .ancestors(TestBaseRule.class)
          .build();
    }
  }

  @Test
  public void testRuleClassTransition() throws Exception {
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new AttributeTransitionRule(),
        new RuleClassTransitionRule());
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
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new AttributeTransitionRule(),
        new RuleClassTransitionRule());
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
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new AttributeTransitionRule(),
        new RuleClassTransitionRule());
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
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new EmptySplitRule(),
        new RuleClassTransitionRule());
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
        new TestAspects.BaseRule(), new TestBaseRule(), new RuleClassTransitionRule());
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
        new TestBaseRule(),
        new RuleClassTransitionRule(),
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

  @Test
  public void ruleTransitionFactoryUsesNonconfigurableAttributesToGenerateTransition()
      throws Exception {
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new AttributeTransitionRule(),
        new UsesRuleTransitionFactoryRule());
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
    setRulesAvailableInTests(
        new TestAspects.BaseRule(),
        new TestBaseRule(),
        new AttributeTransitionRule(),
        new UsesRuleTransitionFactoryRule());
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
        new TestAspects.BaseRule(), new TestBaseRule(), new UsesRuleTransitionFactoryRule());
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
        new TestAspects.BaseRule(), new TestBaseRule(), new UsesRuleTransitionFactoryRule());
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
        getTargetConfiguration(true));
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

      @Override
      public boolean defaultsToSelf() {
        return false;
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

      @Override
      public boolean defaultsToSelf() {
        return false;
      }
    };
  }

  /**
   * Returns the value of {@link TestConfiguration.TestOptions#testFilter} in the output {@link
   * BuildOptions} the given transition applier returns in its current state.
   */
  private List<String> getTestFilterOptionValue(BuildConfiguration.TransitionApplier applier)
      throws Exception {
    Dependency dep = Iterables.getOnlyElement(
        applier.getDependencies(Label.create("some", "target"), AspectCollection.EMPTY));
    ImmutableList.Builder<String> outValues = ImmutableList.builder();
    for (BuildOptions toOptions : ConfiguredTargetFunction.getDynamicTransitionOptions(
        getTargetConfiguration().getOptions(), dep.getTransition(),
        ruleClassProvider.getAllFragments(), ruleClassProvider, false)) {
      outValues.add(toOptions.get(TestConfiguration.TestOptions.class).testFilter);
    }
    return outValues.build();
  }

  @Test
  public void composedStraightTransitions() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyTransition(newPatchTransition("foo"));
    applier.applyTransition(newPatchTransition("bar"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("foobar");
  }

  @Test
  public void composedStraightTransitionThenSplitTransition() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyTransition(newPatchTransition("foo"));
    applier.split(newSplitTransition("split"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("foosplit1", "foosplit2");
  }

  @Test
  public void composedSplitTransitionThenStraightTransition() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.split(newSplitTransition("split"));
    applier.applyTransition(newPatchTransition("foo"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("split1foo", "split2foo");
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
      assertThat(e).hasMessageThat().contains("dependency edges may apply at most one split");
    }
  }

  /**
   * Returns a new {@link Attribute} definition with the given configurator.
   */
  private static Attribute newAttributeWithConfigurator(
      final Attribute.Configurator<BuildOptions> configurator) {
    return attr("foo_attr", LABEL)
        .allowedRuleClasses(ANY_RULE)
        .allowedFileTypes(FileTypeSet.ANY_FILE)
        .cfg(configurator)
        .build();
  }

  /**
   * Returns a new {@link Attribute.Configurator} that appends a given value to {@link
   * TestConfiguration.TestOptions#testFilter}.
   */
  private static Attribute.Configurator<BuildOptions> newAttributeWithStaticConfigurator(
      final String value) {
    return (Attribute.Configurator<BuildOptions>) newAttributeWithConfigurator(
        new Attribute.Configurator<BuildOptions>() {
          @Override
          public Attribute.Transition apply(BuildOptions fromOptions) {
            return newPatchTransition(value);
          }
        }).getConfigurator();
  }

  @Test
  public void attributeConfigurator() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyAttributeConfigurator(newAttributeWithStaticConfigurator("from attr"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("from attr");
  }

  @Test
  public void straightTransitionThenAttributeConfigurator() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyTransition(newPatchTransition("from patch "));
    applier.applyAttributeConfigurator(newAttributeWithStaticConfigurator("from attr"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("from patch from attr");
  }

  /**
   * Returns an {@link Attribute.Configurator} that repeats the existing value of {@link
   * TestConfiguration.TestOptions#testFilter}, plus a signature suffix.
   */
  private static final Attribute.Configurator<BuildOptions> ATTRIBUTE_WITH_REPEATING_CONFIGURATOR =
      (Attribute.Configurator<BuildOptions>)
          newAttributeWithConfigurator(
                  new Attribute.Configurator<BuildOptions>() {
                    @Override
                    public Attribute.Transition apply(BuildOptions fromOptions) {
                      return newPatchTransition(
                          fromOptions.get(TestConfiguration.TestOptions.class).testFilter
                              + " (attr)");
                    }
                  })
              .getConfigurator();

  @Test
  public void splitTransitionThenAttributeConfigurator() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.split(newSplitTransition(" split"));
    applier.applyAttributeConfigurator(ATTRIBUTE_WITH_REPEATING_CONFIGURATOR);
    assertThat(getTestFilterOptionValue(applier))
        .containsExactly(" split1 split1 (attr)", " split2 split2 (attr)");
  }

  @Test
  public void composedAttributeConfigurators() throws Exception {
    update(); // Creates the target configuration.
    BuildConfiguration.TransitionApplier applier = getTargetConfiguration().getTransitionApplier();
    applier.applyAttributeConfigurator(newAttributeWithStaticConfigurator("from attr 1 "));
    applier.applyAttributeConfigurator(newAttributeWithStaticConfigurator("from attr 2"));
    assertThat(getTestFilterOptionValue(applier)).containsExactly("from attr 1 from attr 2");
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
                          .add(MockRule.DEPS_ATTRIBUTE)
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
