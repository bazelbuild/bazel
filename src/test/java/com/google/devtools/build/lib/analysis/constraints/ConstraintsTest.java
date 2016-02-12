// Copyright 2015 The Bazel Authors. All rights reserved.
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
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;

import com.google.devtools.build.lib.analysis.BaseRuleClasses;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.RuleDefinition;
import com.google.devtools.build.lib.analysis.RuleDefinitionEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.RuleClass;
import com.google.devtools.build.lib.packages.RuleClass.Builder;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.testutil.UnknownRuleConfiguredTarget;
import com.google.devtools.build.lib.util.FileTypeSet;

import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

import java.util.Set;

/**
 * Tests for the constraint enforcement system.
 */
@RunWith(JUnit4.class)
public class ConstraintsTest extends AbstractConstraintsTest {

  @Before
  public final void createBuildFile() throws Exception {
    // Support files for RuleClassWithImplicitAndLateBoundDefaults:
    scratch.file("helpers/BUILD",
        "sh_library(name = 'implicit', srcs = ['implicit.sh'])",
        "sh_library(name = 'latebound', srcs = ['latebound.sh'])",
        "sh_library(name = 'default', srcs = ['default.sh'])");
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
          .compatibleWith(env.getLabel("//buildenv/rule_class_compat:b"))
          .restrictedTo(env.getLabel("//buildenv/rule_class_restrict:d"))
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
          .compatibleWith(env.getLabel("//buildenv/rule_class_compat:a"))
          .restrictedTo(env.getLabel("//buildenv/rule_class_compat:b"))
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
          .add(Attribute.attr("$implicit", BuildType.LABEL)
              .value(Label.parseAbsoluteUnchecked("//helpers:implicit")))
          .add(Attribute.attr(":latebound", BuildType.LABEL)
              .value(
                  new Attribute.LateBoundLabel<BuildConfiguration>() {
                    @Override
                    public Label getDefault(Rule rule, AttributeMap attributes,
                        BuildConfiguration configuration) {
                      return Label.parseAbsoluteUnchecked("//helpers:latebound");
                    }
                  }))
          .add(Attribute.attr("normal", BuildType.LABEL)
              .allowedFileTypes(FileTypeSet.NO_FILE)
              .value(Label.parseAbsoluteUnchecked("//helpers:default")))
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

  private static final class RuleClassWithEnforcedImplicitAttribute implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .add(Attribute.attr("$implicit", BuildType.LABEL)
              .value(Label.parseAbsoluteUnchecked("//helpers:implicit"))
              .checkConstraints())
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("rule_with_enforced_implicit_deps")
          .ancestors(BaseRuleClasses.RuleBase.class)
          .factoryClass(UnknownRuleConfiguredTarget.class)
          .build();
    }
  }

  private static final class ConstraintExemptRuleClass implements RuleDefinition {
    @Override
    public RuleClass build(Builder builder, RuleDefinitionEnvironment env) {
      return builder
          .setUndocumented()
          .exemptFromConstraintChecking("for testing removal of restricted_to / compatible_with")
          .build();
    }

    @Override
    public Metadata getMetadata() {
      return RuleDefinition.Metadata.builder()
          .name("totally_free_rule")
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
    builder.addRuleDefinition(new RuleClassWithEnforcedImplicitAttribute());
    builder.addRuleDefinition(new ConstraintExemptRuleClass());
    return builder.build();
  }

  /**
   * Writes the environments and environment groups referred to by the rule class defaults.
   */
  private void writeRuleClassDefaultEnvironments() throws Exception {
    new EnvironmentGroupMaker("buildenv/rule_class_compat").setEnvironments("a", "b")
        .setDefaults("a").make();
    new EnvironmentGroupMaker("buildenv/rule_class_restrict").setEnvironments("c", "d")
        .setDefaults("c").make();
  }

  /**
   * By default, a rule *implicitly* supports all defaults, meaning the explicitly known
   * environment set is empty.
   */
  @Test
  public void testDefaultSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    String ruleDef = getDependencyRule();
    assertThat(supportedEnvironments("dep", ruleDef)).isEmpty();
  }

  /**
   * "Constraining" a rule's environments explicitly sets them.
   */
  @Test
  public void testConstrainedSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    String ruleDef = getDependencyRule(constrainedTo("//buildenv/foo:c"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(asLabelSet("//buildenv/foo:c"));
  }

  /**
   * Specifying compatibility adds the specified environments to the defaults.
   */
  @Test
  public void testCompatibleSupportedEnvironments() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    String ruleDef = getDependencyRule(compatibleWith("//buildenv/foo:c"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(asLabelSet("//buildenv/foo:a", "//buildenv/foo:c"));
  }

  /**
   * A rule can't support *no* environments.
   */
  @Test
  public void testSupportedEnvironmentsConstrainedtoNothing() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(constrainedTo());
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent("attribute cannot be empty");
  }

  /**
   * Restrict the environments within one group, declare compatibility for another.
   */
  @Test
  public void testSupportedEnvironmentsInMultipleGroups() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    new EnvironmentGroupMaker("buildenv/bar").setEnvironments("c", "d").setDefaults("c").make();
    String ruleDef = getDependencyRule(
        constrainedTo("//buildenv/foo:b"), compatibleWith("//buildenv/bar:d"));
    assertThat(supportedEnvironments("dep", ruleDef))
        .containsExactlyElementsIn(
            asLabelSet("//buildenv/foo:b", "//buildenv/bar:c", "//buildenv/bar:d"));
  }

  /**
   * The same label can't appear in both a constraint and a compatibility declaration.
   */
  @Test
  public void testSameEnvironmentCompatibleAndRestricted() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(
        constrainedTo("//buildenv/foo:b"), compatibleWith("//buildenv/foo:b"));
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent("//buildenv/foo:b cannot appear both here and in restricted_to");
  }

  /**
   * Two labels from the same group can't appear in different attributes.
   */
  @Test
  public void testSameGroupCompatibleAndRestricted() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    reporter.removeHandler(failFastHandler);
    String ruleDef = getDependencyRule(
        constrainedTo("//buildenv/foo:a"), compatibleWith("//buildenv/foo:b"));
    assertNull(scratchConfiguredTarget("hello", "dep", ruleDef));
    assertContainsEvent(
        "//buildenv/foo:b and //buildenv/foo:a belong to the same environment group");
  }

  /**
   * Tests that rule class defaults change a rule's default set of environments.
   */
  @Test
  public void testSupportedEnvironmentsRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    String ruleDef = "rule_class_default(name = 'a')";
    Set<Label> expectedEnvironments = asLabelSet("//buildenv/rule_class_compat:a",
        "//buildenv/rule_class_compat:b", "//buildenv/rule_class_restrict:d");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that explicit declarations override rule class defaults.
   */
  @Test
  public void testExplicitAttributesOverrideRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    String ruleDef = "rule_class_default("
        + "    name = 'a',"
        + "    compatible_with = ['//buildenv/rule_class_restrict:c'],"
        + "    restricted_to = ['//buildenv/rule_class_compat:a'],"
        + ")";
    Set<Label> expectedEnvironments = asLabelSet("//buildenv/rule_class_compat:a",
        "//buildenv/rule_class_restrict:c", "//buildenv/rule_class_restrict:d");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that a rule's "known" supported environments includes those from groups referenced
   * in rule class defaults but not in explicit rule attributes.
   */
  @Test
  public void testKnownEnvironmentsIncludesThoseFromRuleClassDefaults() throws Exception {
    writeRuleClassDefaultEnvironments();
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    String ruleDef = "rule_class_default("
        + "    name = 'a',"
        + "    restricted_to = ['//buildenv/foo:b'],"
        + ")";
    Set<Label> expectedEnvironments = asLabelSet("//buildenv/rule_class_compat:a",
        "//buildenv/rule_class_compat:b", "//buildenv/rule_class_restrict:d",
        "//buildenv/foo:b");
    assertThat(supportedEnvironments("a", ruleDef)).containsExactlyElementsIn(expectedEnvironments);
  }

  /**
   * Tests that environments from the same group can't appear in both restriction and
   * compatibility rule class defaults.
   */
  @Test
  public void testSameEnvironmentRuleClassCompatibleAndRestricted() throws Exception {
    writeRuleClassDefaultEnvironments();
    reporter.removeHandler(failFastHandler);
    String ruleDef = "bad_rule_class_default(name = 'a')";
    assertNull(scratchConfiguredTarget("hello", "a", ruleDef));
    assertContainsEvent("//buildenv/rule_class_compat:a and //buildenv/rule_class_compat:b "
        + "belong to the same environment group");
  }

  /**
   * Tests that a dependency is valid if both rules implicitly inherit all default environments.
   */
  @Test
  public void testAllDefaults() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both rules explicitly declare the same constraints.
   */
  @Test
  public void testSameConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:b")),
        getDependingRule(constrainedTo("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both the depender and dependency explicitly declare
   * their constraints and the depender supports a subset of the dependency's environments
   */
  @Test
  public void testValidConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:a", "//buildenv/foo:b")),
        getDependingRule(constrainedTo("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is invalid when both the depender and dependency explicitly declare
   * their constraints and the depender supports an environment the dependency doesn't.
   */
  @Test
  public void testInvalidConstraintsDeclaredExplicitly() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:b")),
        getDependingRule(constrainedTo("//buildenv/foo:a", "//buildenv/foo:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:a");
  }

  /**
   * Tests that a dependency is valid when both rules add the same set of environments to their
   * defaults.
   */
  @Test
  public void testSameCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")),
        getDependingRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is valid when both rules add environments to their defaults and
   * the depender only adds environments also added by the dependency.
   */
  @Test
  public void testValidCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")),
        getDependingRule(compatibleWith("//buildenv/foo:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a dependency is invalid when both rules add environments to their defaults and
   * the depender adds environments not added by the dependency.
   */
  @Test
  public void testInvalidCompatibilityConstraints() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(compatibleWith("//buildenv/foo:c")),
        getDependingRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:b");
  }

  /**
   * Tests the error message when the dependency is missing multiple expected environments.
   */
  @Test
  public void testMultipleMissingEnvironments() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent("dependency //hello:dep doesn't support expected environments: "
        + "//buildenv/foo:b, //buildenv/foo:c");
  }

  /**
   * Tests a valid dependency including environments from different groups.
   */
  @Test
  public void testValidMultigroupConstraints() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    new EnvironmentGroupMaker("buildenv/bar").setEnvironments("d", "e", "f").setDefaults("d")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:b", "//buildenv/foo:c"),
            compatibleWith("//buildenv/bar:e")),
        getDependingRule(constrainedTo("//buildenv/foo:c"), compatibleWith("//buildenv/bar:e")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency including environments from different groups.
   */
  @Test
  public void testInvalidMultigroupConstraints() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    new EnvironmentGroupMaker("buildenv/bar").setEnvironments("d", "e", "f").setDefaults("d")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:c"), compatibleWith("//buildenv/bar:e")),
        getDependingRule(constrainedTo("//buildenv/foo:b", "//buildenv/foo:c"),
            compatibleWith("//buildenv/bar:e")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:b");
  }

  /**
   * Tests a valid dependency where the dependency doesn't "know" about the expected environment's
   * group, but implicitly supports it because that environment is a default.
   */
  @Test
  public void testValidConstraintsUnknownEnvironmentToDependency() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a", "b")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency where the dependency doesn't "know" about the expected
   * environment's group and doesn't support it because it isn't a default.
   */
  @Test
  public void testInvalidConstraintsUnknownEnvironmentToDependency() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a", "b")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//buildenv/foo:c")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:c");
  }

  /**
   * Tests a valid dependency where the depender doesn't "know" about one of the dependency's
   * groups, the depender implicitly supports that group's defaults, and all of those defaults
   * are accounted for in the dependency.
   */
  @Test
  public void testValidConstraintsUnknownEnvironmentToDependender() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:a", "//buildenv/foo:b")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests an invalid dependency where the depender doesn't "know" about one of the dependency's
   * groups, the depender implicitly supports that group's defaults, and one of those defaults
   * isn't accounted for in the dependency.
   */
  @Test
  public void testInvalidConstraintsUnknownEnvironmentToDependender() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:b")),
        getDependingRule());
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:a");
  }

  /**
   * Tests the case where one dependency is valid and another one isn't.
   */
  @Test
  public void testOneDependencyIsInvalid() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        getRuleDef("sh_library", "bad_dep", constrainedTo("//buildenv/foo:b")),
        getRuleDef("sh_library", "good_dep", compatibleWith("//buildenv/foo:b")),
        getRuleDef("sh_library", "depender",
            constrainedTo("//buildenv/foo:a", "//buildenv/foo:b"),
            getAttrDef("deps", "good_dep", "bad_dep")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:depender"));
    assertContainsEvent("//hello:bad_dep doesn't support expected environment: //buildenv/foo:a");
    assertDoesNotContainEvent("//hello:good_dep");
  }

  @Test
  public void testConstraintEnforcementDisabled() throws Exception {
    useConfiguration("--experimental_enforce_constraints=0");
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b", "c").setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(compatibleWith("//buildenv/foo:b", "//buildenv/foo:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that package defaults compatibility produces a valid dependency that would otherwise
   * be invalid.
   */
  @Test
  public void testCompatibilityPackageDefaults() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        "package(default_compatible_with = ['//buildenv/foo:b'])",
        getDependencyRule(),
        getDependingRule(compatibleWith("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a rule's compatibility declaration overrides its package defaults compatibility.
   */
  @Test
  public void testPackageDefaultsCompatibilityOverride() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults().make();
    // We intentionally create an invalid dependency structure vs. a valid one. If we tested on
    // a valid one, this test wouldn't be able to distinguish between rule declarations overriding
    // package defaults and package defaults overriding rule declarations.
    scratch.file("hello/BUILD",
        "package(default_compatible_with = ['//buildenv/foo:b'])",
        getDependencyRule(compatibleWith("//buildenv/foo:a")),
        getDependingRule(compatibleWith("//buildenv/foo:a", "//buildenv/foo:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:b");
  }

  /**
   * Tests that package defaults restriction produces an valid dependency that would otherwise
   * be invalid.
   */
  @Test
  public void testRestrictionPackageDefaults() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a", "b")
        .make();
    scratch.file("hello/BUILD",
        "package(default_restricted_to = ['//buildenv/foo:b'])",
        getDependencyRule(constrainedTo("//buildenv/foo:b")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  /**
   * Tests that a rule's restriction declaration overrides its package defaults restriction.
   */
  @Test
  public void testPackageDefaultsRestrictionOverride() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults().make();
    // We intentionally create an invalid dependency structure vs. a valid one. If we tested on
    // a valid one, this test wouldn't be able to distinguish between rule declarations overriding
    // package defaults and package defaults overriding rule declarations.
    scratch.file("hello/BUILD",
        "package(default_restricted_to = ['//buildenv/foo:b'])",
        getDependencyRule(constrainedTo("//buildenv/foo:a")),
        getDependingRule(constrainedTo("//buildenv/foo:a", "//buildenv/foo:b")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:b");
  }

  /**
   * Tests that "default_compatible_with" fills in a rule's "compatible_with" when not specified
   * by the rule. This is different than, e.g., the rule declaration / rule class defaults model,
   * where the "compatible_with" / "restricted_to" values of rule class defaults are merged together
   * before being supplied to the rule. See comments in DependencyResolver for more discussion.
   */
  @Test
  public void testPackageDefaultsDirectlyFillRuleAttributes() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults().make();
    scratch.file("hello/BUILD",
        "package(default_restricted_to = ['//buildenv/foo:b'])",
        getDependencyRule(compatibleWith("//buildenv/foo:a")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:dep"));
    assertContainsEvent("//buildenv/foo:a and //buildenv/foo:b belong to the same "
        + "environment group. They should be declared together either here or in restricted_to");
  }

  @Test
  public void testHostDependenciesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        "sh_binary(name = 'host_tool',",
        "    srcs = ['host_tool.sh'],",
        "    restricted_to = ['//buildenv/foo:b'])",
        "genrule(",
        "    name = 'hello',",
        "    srcs = [],",
        "    outs = ['hello.out'],",
        "    cmd = '',",
        "    tools = [':host_tool'],",
        "    compatible_with = ['//buildenv/foo:a'])");
    assertNotNull(getConfiguredTarget("//hello:hello"));
    assertNoEvents();
  }

  @Test
  public void testHostDependenciesNotCheckedNoDistinctHostConfiguration() throws Exception {
    useConfiguration("--nodistinct_host_configuration");
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        "sh_binary(name = 'host_tool',",
        "    srcs = ['host_tool.sh'],",
        "    restricted_to = ['//buildenv/foo:b'])",
        "genrule(",
        "    name = 'hello',",
        "    srcs = [],",
        "    outs = ['hello.out'],",
        "    cmd = '',",
        "    tools = [':host_tool'],",
        "    compatible_with = ['//buildenv/foo:a'])");
    assertNotNull(getConfiguredTarget("//hello:hello"));
    assertNoEvents();
  }

  @Test
  public void testImplicitAndLateBoundDependenciesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        "rule_with_implicit_and_latebound_deps(",
        "    name = 'hi',",
        "    compatible_with = ['//buildenv/foo:b'])");
    assertNotNull(getConfiguredTarget("//hello:hi"));
    // Note that the event "cannot build rule_with_implicit_and_latebound_deps" *does* occur
    // because of the implementation of UnknownRuleConfiguredTarget.
    assertDoesNotContainEvent(":implicit doesn't support expected environment");
    assertDoesNotContainEvent(":latebound doesn't support expected environment");
    assertDoesNotContainEvent("normal doesn't support expected environment");
  }

  @Test
  public void testImplicitDepsWithWhiteListedAttributeAreChecked() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults("a").make();
    scratch.file("hello/BUILD",
        "rule_with_enforced_implicit_deps(",
        "    name = 'hi',",
        "    compatible_with = ['//buildenv/foo:b'])");
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:hi"));
    assertContainsEvent(
        "dependency //helpers:implicit doesn't support expected environment: //buildenv/foo:b");
  }

  @Test
  public void testOutputFilesAreChecked() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults().make();
    scratch.file("hello/BUILD",
        "genrule(name = 'gen', srcs = [], outs = ['shlib.sh'], cmd = '')",
        "sh_library(",
        "    name = 'shlib',",
        "    srcs = ['shlib.sh'],",
        "    data = ['whatever.txt'],",
        "    compatible_with = ['//buildenv/foo:a'])");
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:shlib"));
    assertContainsEvent(
        "dependency //hello:gen doesn't support expected environment: //buildenv/foo:a");
  }

  @Test
  public void testConfigSettingRulesAreNotChecked() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo").setEnvironments("a", "b").setDefaults().make();
    scratch.file("hello/BUILD",
        "config_setting(name = 'setting', values = {'compilation_mode': 'fastbuild'})",
        "sh_library(",
        "    name = 'shlib',",
        "    srcs = select({",
        "        ':setting': ['shlib.sh'],",
        "    }),",
        "    compatible_with = ['//buildenv/foo:a'])");
    assertNotNull(getConfiguredTarget("//hello:shlib"));
    assertNoEvents();
  }

  @Test
  public void testFulfills() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults()
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:a")),
        getDependingRule(constrainedTo("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  @Test
  public void testFulfillsIsNotSymmetric() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults()
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:b")),
        getDependingRule(constrainedTo("//buildenv/foo:a")));
    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//hello:main"));
    assertContainsEvent(
        "dependency //hello:dep doesn't support expected environment: //buildenv/foo:a");
  }

  @Test
  public void testFulfillsIsTransitive() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b", "c")
        .setFulfills("a", "b")
        .setFulfills("b", "c")
        .setDefaults()
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:a")),
        getDependingRule(constrainedTo("//buildenv/foo:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  @Test
  public void testDefaultEnvironmentDirectlyFulfills() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//buildenv/foo:b")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  @Test
  public void testDefaultEnvironmentIndirectlyFulfills() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b", "c")
        .setFulfills("a", "b")
        .setFulfills("b", "c")
        .setDefaults("a")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(),
        getDependingRule(constrainedTo("//buildenv/foo:c")));
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  @Test
  public void testEnvironmentFulfillsExpectedDefault() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setFulfills("a", "b")
        .setDefaults("b")
        .make();
    scratch.file("hello/BUILD",
        getDependencyRule(constrainedTo("//buildenv/foo:a")),
        getDependingRule());
    assertNotNull(getConfiguredTarget("//hello:main"));
    assertNoEvents();
  }

  @Test
  public void testConstraintExemptRulesDontHaveConstraintAttributes() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setDefaults("a")
        .make();
    scratch.file("ihave/BUILD",
        "totally_free_rule(",
        "    name = 'nolimits',",
        "    restricted_to = ['//buildenv/foo:b']",
        ")");

    reporter.removeHandler(failFastHandler);
    assertNull(getConfiguredTarget("//ihave:nolimits"));
    assertContainsEvent("no such attribute 'restricted_to' in 'totally_free_rule'");
  }

  @Test
  public void testBuildingEnvironmentGroupDirectlyDoesntCrash() throws Exception {
    new EnvironmentGroupMaker("buildenv/foo")
        .setEnvironments("a", "b")
        .setDefaults("a")
        .make();
    assertNotNull(getConfiguredTarget("//buildenv/foo:foo"));
  }
}
