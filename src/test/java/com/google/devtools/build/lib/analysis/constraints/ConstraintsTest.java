// Copyright 2015 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.analysis.constraints;

import static com.google.common.truth.Truth.assertThat;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.syntax.Label;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.UnknownRuleConfiguredTarget;

import java.util.Set;

/**
 * Tests for the constraint enforcement system.
 */
public class ConstraintsTest extends AbstractConstraintsTest {

  @Override
  public void setUp() throws Exception {
    super.setUp();
    // Support files for RuleClassWithImplicitAndLateBoundDefaults:
    scratchFile("helpers/BUILD",
        "sh_library(name = 'implicit', srcs = ['implicit.sh'])",
        "sh_library(name = 'latebound', srcs = ['latebound.sh'])");
  }

  /**
   * Dummy rule class for testing rule class defaults. This class applies valid defaults. Note
   * that the specified environments must be independently created.
   */
  private static final class RuleClassDefaultRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .compatibleWith(env.getLabel("//rule_class_compat:b"))
          .restrictedTo(env.getLabel("//rule_class_restrict:d"))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_class_default")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  /**
   * Dummy rule class for testing rule class defaults. This class applies invalid defaults. Note
   * that the specified environments must be independently created.
   */
  private static final class BadRuleClassDefaultRule implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          // These defaults are invalid since compatibleWith and restrictedTo can't mix
          // environments from the same group.
          .compatibleWith(env.getLabel("//rule_class_compat:a"))
          .restrictedTo(env.getLabel("//rule_class_compat:b"))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("bad_rule_class_default")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  private static final class RuleClassWithImplicitAndLateBoundDefaults implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .add(Attribute.attr("$implicit", Type.LABEL)
              .value(Label.parseAbsoluteUnchecked("//helpers:implicit")))
          .add(Attribute.attr(":latebound", Type.LABEL)
              .value(
                  new Attribute.LateBoundLabel<BuildConfiguration>() {
                    @Override
                    public Label getDefault(Rule rule, BuildConfiguration configuration) {
                      return Label.parseAbsoluteUnchecked("//helpers:latebound");
                    }
                  }))
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_with_implicit_and_latebound_deps")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  /**
   * Injects the rule class default rules into the default test rule class provider.
   */
  @Override
  protected ConfiguredRuleClassProvider getRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder = new ConfiguredRuleClassProvider.Builder();
    TestRuleClassProvider.addStandardRules(builder);
    builder.addRuleDefinition(new RuleClassDefaultRule());
    builder.addRuleDefinition(new BadRuleClassDefaultRule());
    builder.addRuleDefinition(new RuleClassWithImplicitAndLateBoundDefaults());
    return builder.build();
  }

  /**
   * Writes the environments and environment groups referred to by the rule class defaults.
   */
  private void writeRuleClassDefaultEnvironments() throws Exception {
    new EnvironmentGroupMaker("rule_class_compat").setEnvironments("a", "b").setDefaults("a")
        .make();
    new EnvironmentGroupMaker("rule_class_restrict").setEnvironments("c", "d").setDefaults("c")
        .make();
  }

  /**
   * By default, a rule *implicitly* supports all defaults, meaning the explicitly known
   * environment set is empty.
   */
  public void testDefaultSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    String ruleDef = getDependencyRule();
    assertThat(supportedEnvironments("dep", ruleDef)).isEmpty();
  }

  /**
   * "Constraining" a rule's environments explicitly sets them.
   */
  public void testConstrainedSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    String ruleDef = getDependencyRule(constrainedTo("//foo_env:c"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(asLabelSet("//foo_env:c"));
  }

  /**
   * Specifying compatibility adds the specified environments to the defaults.
   */
  public void testCompatibleSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    String ruleDef = getDependencyRule(compatibleWith("//foo_env:c"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(asLabelSet("//foo_env:a", "//foo_env:c"));
  }

  /**
   * A rule can't support *no* environments.
   */
  public void testSupportedEnvironmentsConstrainedtoNothing() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(constrainedTo());
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent("attribute cannot be empty");
  }

  /**
   * Restrict the environments within one group, declare compatibility for another.
   */
  public void testSupportedEnvironmentsInMultipleGroups() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    new EnvironmentGroupMaker("bar_env").setEnvironments("c", "d").setDefaults("c").make();
    String ruleDef = getDependencyRule(constrainedTo("//foo_env:b"), compatibleWith("//bar_env:d"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(asLabelSet("//foo_env:b", "//bar_env:c", "//bar_env:d"));
  }

  /**
   * The same label can't appear in both a constraint and a compatibility declaration.
   */
  public void testSameEnvironmentCompatibleAndRestricted() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(constrainedTo("//foo_env:b"), compatibleWith("//foo_env:b"));
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent("//foo_env:b cannot appear both here and in restricted_to");
  }

  /**
   * Two labels from the same group can't appear in different attributes.
   */
  public void testSameGroupCompatibleAndRestricted() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(constrainedTo("//foo_env:a"), compatibleWith("//foo_env:b"));
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent("//foo_env:b and //foo_env:a belong to the same environment group");
  }

  /**
   * Tests that rule class defaults change a rule's default set of environments.
   */
  public void testSupportedEnvironmentsRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    String ruleDef = "rule_class_default(name = 'a')";
    Set<Label> expectedEnvironments =
        asLabelSet("//rule_class_compat:a", "//rule_class_compat:b", "//rule_class_restrict:d");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that explicit declarations override rule class defaults.
   */
  public void testExplicitAttributesOverrideRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    String ruleDef = "rule_class_default("
        + "    name = 'a',"
        + "    compatible_with = ['//rule_class_restrict:c'],"
        + "    restricted_to = ['//rule_class_compat:a'],"
        + ")";
    Set<Label> expectedEnvironments =
        asLabelSet("//rule_class_compat:a", "//rule_class_restrict:c", "//rule_class_restrict:d");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that a rule's "known" supported environments includes those from groups referenced
   * in rule class defaults but not in explicit rule attributes.
   */
  public void testKnownEnvironmentsIncludesThoseFromRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    String ruleDef = "rule_class_default("
        + "    name = 'a',"
        + "    restricted_to = ['//foo_env:b'],"
        + ")";
    Set<Label> expectedEnvironments = asLabelSet("//rule_class_compat:a", "//rule_class_compat:b",
        "//rule_class_restrict:d", "//foo_env:b");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that environments from the same group can't appear in both restriction and
   * compatibility rule class defaults.
   */
  public void testSameEnvironmentRuleClassCompatibleAndRestricted() throws Exception {
    writeRuleClassDefaultEnvironments();
    reporter.removeHandler(failFastHandler);
    String ruleDef = "bad_rule_class_default(name = 'a')";
    assertNull(scratchConfiguredTarget("hello", "a", ruleDef));
    assertContainsEvent(
        "//rule_class_compat:a and //rule_class_compat:b belong to the same environment group");
  }

  /**
   * Tests that a dependency is valid if both rules implicitly inherit all default environments.
   */
  public void testAllDefaults() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both rules explicitly declare the same constraints.
   */
  public void testSameConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:b")),
        getDependingRule(constrainedTo("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both the depender and dependency explicitly declare
   * their constraints and the depender supports a subset of the dependency's environments
   */
  public void testValidConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:a", "//foo_env:b")),
        getDependingRule(constrainedTo("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is invalid when both the depender and dependency explicitly declare
   * their constraints and the depender supports an environment the dependency doesn't.
   */
  public void testInvalidConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:b")),
        getDependingRule(constrainedTo("//foo_env:a", "//foo_env:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:a");
  }

  /**
   * Tests that a dependency is valid when both rules add the same set of environments to their
   * defaults.
   */
  public void testSameCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(compatibleWith("//foo_env:b", "//foo_env:c")),
        getDependingRule(compatibleWith("//foo_env:b", "//foo_env:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both rules add environments to their defaults and
   * the depender only adds environments also added by the dependency.
   */
  public void testValidCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(compatibleWith("//foo_env:b", "//foo_env:c")),
        getDependingRule(compatibleWith("//foo_env:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is invalid when both rules add environments to their defaults and
   * the depender adds environments not added by the dependency.
   */
  public void testInvalidCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(compatibleWith("//foo_env:c")),
        getDependingRule(compatibleWith("//foo_env:b", "//foo_env:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:b");
  }

  /**
   * Tests the error message when the dependency is missing multiple expected environments.
   */
  public void testMultipleMissingEnvironments() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(compatibleWith("//foo_env:b", "//foo_env:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environments: //foo_env:b, //foo_env:c");
  }

  /**
   * Tests a valid dependency including environments from different groups.
   */
  public void testValidMultigroupConstraints() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    new EnvironmentGroupMaker("bar_env").setEnvironments("d", "e", "f").setDefaults("d").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:b", "//foo_env:c"),
            compatibleWith("//bar_env:e")),
        getDependingRule(constrainedTo("//foo_env:c"), compatibleWith("//bar_env:e")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency including environments from different groups.
   */
  public void testInvalidMultigroupConstraints() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    new EnvironmentGroupMaker("bar_env").setEnvironments("d", "e", "f").setDefaults("d").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:c"), compatibleWith("//bar_env:e")),
        getDependingRule(constrainedTo("//foo_env:b", "//foo_env:c"),
            compatibleWith("//bar_env:e")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:b");
  }

  /**
   * Tests a valid dependency where the dependency doesn't "know" about the expected environment's
   * group, but implicitly supports it because that environment is a default.
   */
  public void testValidConstraintsUnknownEnvironmentToDependency() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a", "b")
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency where the dependency doesn't "know" about the expected
   * environment's group and doesn't support it because it isn't a default.
   */
  public void testInvalidConstraintsUnknownEnvironmentToDependency() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a", "b")
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//foo_env:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:c");
  }

  /**
   * Tests a valid dependency where the depender doesn't "know" about one of the dependency's
   * groups, the depender implicitly supports that group's defaults, and all of those defaults
   * are accounted for in the dependency.
   */
  public void testValidConstraintsUnknownEnvironmentToDependender() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:a", "//foo_env:b")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency where the depender doesn't "know" about one of the dependency's
   * groups, the depender implicitly supports that group's defaults, and one of those defaults
   * isn't accounted for in the dependency.
   */
  public void testInvalidConstraintsUnknownEnvironmentToDependender() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:b")),
        getDependingRule());
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:a");
  }

  /**
   * Tests the case where one dependency is valid and another one isn't.
   */
  public void testOneDependencyIsInvalid() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getRuleDef("sh_library", "bad_dep", constrainedTo("//foo_env:b")),
        getRuleDef("sh_library", "good_dep", compatibleWith("//foo_env:b")),
        getRuleDef("sh_library", "depender",
            constrainedTo("//foo_env:a", "//foo_env:b"),
            getAttrDef("deps", "good_dep", "bad_dep")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:depender"));
    assertContainsEvent("//hello:bad_dep doesn't support expected environment: //foo_env:a");
    assertDoesNotContainEvent("//hello:good_dep");
  }

  public void testConstraintEnforcementDisabled() throws Exception {
    useConfiguration("--experimental_enforce_constraints=0");
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b", "c").setDefaults("a").make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(compatibleWith("//foo_env:b", "//foo_env:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that package defaults compatibility produces a valid dependency that would otherwise
   * be invalid.
   */
  public void testCompatibilityPackageDefaults() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        "package(default_compatible_with = ['//foo_env:b'])",
        getDependencyRule(),
        getDependingRule(compatibleWith("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a rule's compatibility declaration overrides its package defaults compatibility.
   */
  public void testPackageDefaultsCompatibilityOverride() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults().make();
    // We intentionally create an invalid dependency structure vs. a valid one. If we tested on
    // a valid one, this test wouldn't be able to distinguish between rule declarations overriding
    // package defaults and package defaults overriding rule declarations.
    scratchFile("hello/BUILD",
        "package(default_compatible_with = ['//foo_env:b'])",
        getDependencyRule(compatibleWith("//foo_env:a")),
        getDependingRule(compatibleWith("//foo_env:a", "//foo_env:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:b");
  }

  /**
   * Tests that package defaults restriction produces an valid dependency that would otherwise
   * be invalid.
   */
  public void testRestrictionPackageDefaults() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a", "b").make();
    scratchFile("hello/BUILD",
        "package(default_restricted_to = ['//foo_env:b'])",
        getDependencyRule(constrainedTo("//foo_env:b")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a rule's restriction declaration overrides its package defaults restriction.
   */
  public void testPackageDefaultsRestrictionOverride() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults().make();
    // We intentionally create an invalid dependency structure vs. a valid one. If we tested on
    // a valid one, this test wouldn't be able to distinguish between rule declarations overriding
    // package defaults and package defaults overriding rule declarations.
    scratchFile("hello/BUILD",
        "package(default_restricted_to = ['//foo_env:b'])",
        getDependencyRule(constrainedTo("//foo_env:a")),
        getDependingRule(constrainedTo("//foo_env:a", "//foo_env:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:b");
  }

  /**
   * Tests that "default_compatible_with" fills in a rule's "compatible_with" when not specified
   * by the rule. This is different than, e.g., the rule declaration / rule class defaults model,
   * where the "compatible_with" / "restricted_to" values of rule class defaults are merged together
   * before being supplied to the rule. See comments in DependencyResolver for more discussion.
   */
  public void testPackageDefaultsDirectlyFillRuleAttributes() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults().make();
    scratchFile("hello/BUILD",
        "package(default_restricted_to = ['//foo_env:b'])",
        getDependencyRule(compatibleWith("//foo_env:a")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:dep"));
    assertContainsEvent("//foo_env:a and //foo_env:b belong to the same environment group. They "
        + "should be declared together either here or in restricted_to");
  }

  public void testHostDependenciesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        "sh_binary(name = 'host_tool', srcs = ['host_tool.sh'], restricted_to = ['//foo_env:b'])",
        "genrule(",
        "    name = 'hello',",
        "    srcs = [],",
        "    outs = ['hello.out'],",
        "    cmd = '',",
        "    tools = [':host_tool'],",
        "    compatible_with = ['//foo_env:a'])");
    assertNotNull(getConfiguredTarget("//hello:hello"));
    assertNoEvents();
  }

  public void testImplicitAndLateBoundDependenciesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults("a").make();
    scratchFile("hello/BUILD",
        "rule_with_implicit_and_latebound_deps(name = 'hi', compatible_with = ['//foo_env:b'])");
    assertNotNull(getConfiguredTarget("//hello:hi"));
    // Note that the event "cannot build rule_with_implicit_and_latebound_deps" *does* occur
    // because of the implementation of UnknownRuleConfiguredTarget.
    assertDoesNotContainEvent(":implicit doesn't support expected environment");
    assertDoesNotContainEvent(":latebound doesn't support expected environment");
  }

  public void testOutputFilesAreChecked() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults().make();
    scratchFile("hello/BUILD",
        "genrule(name = 'gen', srcs = [], outs = ['shlib.sh'], cmd = '')",
        "sh_library(",
        "    name = 'shlib',",
        "    srcs = ['shlib.sh'],",
        "    data = ['whatever.txt'],",
        "    compatible_with = ['//foo_env:a'])");
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:shlib"));
    assertContainsEvent("dependency //hello:gen doesn't support expected environment: //foo_env:a");
  }

  public void testConfigSettingRulesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("foo_env").setEnvironments("a", "b").setDefaults().make();
    scratchFile("hello/BUILD",
        "config_setting(name = 'setting', values = {'compilation_mode': 'fastbuild'})",
        "sh_library(",
        "    name = 'shlib',",
        "    srcs = select({",
        "        ':setting': ['shlib.sh'],",
        "    }),",
        "    compatible_with = ['//foo_env:a'])");
    assertNotNull(getConfiguredTarget("//hello:shlib"));
    assertNoEvents();
  }

  public void testFulfills() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults()
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:a")),
        getDependingRule(constrainedTo("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  public void testFulfillsIsNotSymmetric() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults()
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:b")),
        getDependingRule(constrainedTo("//foo_env:a")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environment: //foo_env:a");
  }

  public void testFulfillsIsTransitive() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b", "c")
        .setFulfills("a", "b")
        .setFulfills("b", "c")
        .setDefaults()
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:a")),
        getDependingRule(constrainedTo("//foo_env:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  public void testDefaultEnvironmentDirectlyFulfills() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults("a")
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//foo_env:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  public void testDefaultEnvironmentIndirectlyFulfills() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b", "c")
        .setFulfills("a", "b")
        .setFulfills("b", "c")
        .setDefaults("a")
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//foo_env:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  public void testEnvironmentFulfillsExpectedDefault() throws Exception {
    new EnvironmentGroupMaker("foo_env")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults("b")
        .make();
    scratchFile("hello/BUILD",
        getDependencyRule(constrainedTo("//foo_env:a")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }
}
