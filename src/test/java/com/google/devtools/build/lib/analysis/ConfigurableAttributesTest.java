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
package com.google.devtools.build.lib.analysis;

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.analysis.util.BuildViewTestBase.AnalysisFailureRecorder;
import com.google.devtools.build.lib.analysis.util.BuildViewTestCase;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.Attribute;
import com.google.devtools.build.lib.packages.Attribute.ComputedDefault;
import com.google.devtools.build.lib.packages.AttributeMap;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.RuleClass.ToolchainResolutionMode;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetAndData;
import com.google.devtools.build.lib.testutil.TestRuleClassProvider;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.io.IOException;
import java.util.Collection;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Integration tests for configurable attributes.
 */
@RunWith(JUnit4.class)
public class ConfigurableAttributesTest extends BuildViewTestCase {

  private void writeConfigRules() throws Exception {
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");
  }

  private void writeHelloRules(boolean includeDefaultCondition) throws IOException {
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = ['hello.java'],",
        "    deps = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        includeDefaultCondition
            ? "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],"
            : "",
        "    }))",
        "",
        "java_library(",
        "    name = 'adep',",
        "    srcs = ['adep.java'])",
        "java_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.java'])",
        "java_library(",
        "    name = 'defaultdep',",
        "    srcs = ['defaultdep.java'])");
  }

  private static final String ADEP_INPUT = "bin java/hello/libadep.jar";
  private static final String BDEP_INPUT = "bin java/hello/libbdep.jar";
  private static final String CDEP_INPUT = "bin java/hello/libcdep.jar";
  private static final String DEFAULTDEP_INPUT = "bin java/hello/libdefaultdep.jar";

  /**
   * Checks that, given the specified configuration parameters, the input rule *has* the expected
   * attribute values and *doesn't have* the unexpected attribute values.
   */
  private void checkRule(
      String ruleLabel,
      String attributeName,
      Collection<String> options,
      Iterable<String> expected,
      Iterable<String> notExpected)
      throws Exception {
    useConfiguration(options.toArray(new String[options.size()]));
    ConfiguredTarget binary = getConfiguredTarget(ruleLabel);
    assertThat(binary).isNotNull();
    Set<String> actualDeps = artifactsToStrings(getPrerequisiteArtifacts(binary, attributeName));
    expected.forEach(expectedInput -> assertThat(actualDeps).contains(expectedInput));
    notExpected.forEach(unexpectedInput -> assertThat(actualDeps).doesNotContain(unexpectedInput));
  }

  private void checkRule(String ruleLabel, String option,
      Iterable<String> expected, Iterable<String> notExpected) throws Exception {
    checkRule(ruleLabel, ImmutableList.of(option), expected, notExpected);
  }

  private void checkRule(
      String ruleLabel,
      Collection<String> options,
      Iterable<String> expected,
      Iterable<String> notExpected)
      throws Exception {
    checkRule(ruleLabel, "deps", options, expected, notExpected);
  }

  private static final MockRule RULE_WITH_OUTPUT_ATTR =
      () -> MockRule.define("rule_with_output_attr", attr("out", BuildType.OUTPUT));

  private static final MockRule RULE_WITH_COMPUTED_DEFAULT =
      () -> MockRule.define(
          "rule_with_computed_default",
          attr("string_attr", Type.STRING),
          attr("$computed_attr", Type.STRING).value(
              new ComputedDefault("string_attr") {
                @Override
                public Object getDefault(AttributeMap rule) {
                  return rule.get("string_attr", Type.STRING) + "2";
                }
            }));

  private static final MockRule RULE_WITH_BOOLEAN_ATTR =
      () -> MockRule.define("rule_with_boolean_attr", attr("boolean_attr", Type.BOOLEAN));

  private static final MockRule RULE_WITH_ALLOWED_VALUES =
      () -> MockRule.define(
          "rule_with_allowed_values",
          attr("one_two", Type.STRING)
              .allowedValues(new Attribute.AllowedValueSet("one", "two")));

  private static final MockRule RULE_WITH_LABEL_DEFAULT =
      () ->
          MockRule.define(
              "rule_with_label_default",
              (builder, env) ->
                  builder.add(
                      attr("dep", BuildType.LABEL)
                          .value(Label.parseAbsoluteUnchecked("//foo:default"))
                          .allowedFileTypes(FileTypeSet.ANY_FILE)));

  private static final MockRule RULE_WITH_NO_PLATFORM =
      () ->
          MockRule.define(
              "rule_with_no_platform",
              (builder, env) ->
                  builder
                      .add(attr("deps", LABEL_LIST).allowedFileTypes())
                      .useToolchainResolution(ToolchainResolutionMode.DISABLED));

  @Override
  protected ConfiguredRuleClassProvider createRuleClassProvider() {
    ConfiguredRuleClassProvider.Builder builder =
        new ConfiguredRuleClassProvider.Builder()
            .addRuleDefinition(RULE_WITH_OUTPUT_ATTR)
            .addRuleDefinition(RULE_WITH_COMPUTED_DEFAULT)
            .addRuleDefinition(RULE_WITH_BOOLEAN_ATTR)
            .addRuleDefinition(RULE_WITH_ALLOWED_VALUES)
            .addRuleDefinition(RULE_WITH_LABEL_DEFAULT)
            .addRuleDefinition(RULE_WITH_NO_PLATFORM);
    TestRuleClassProvider.addStandardRules(builder);
    return builder.build();
  }

  @Test
  public void basicConfigurability() throws Exception {
    writeHelloRules(/*includeDefaultCondition=*/true);
    writeConfigRules();
    checkRule("//java/hello:hello", "--test_arg=a",
        /*expected:*/ ImmutableList.of(ADEP_INPUT),
        /*not expected:*/ ImmutableList.of(BDEP_INPUT, DEFAULTDEP_INPUT));
    checkRule("//java/hello:hello", "--test_arg=b",
        /*expected:*/ ImmutableList.of(BDEP_INPUT),
        /*not expected:*/ ImmutableList.of(ADEP_INPUT, DEFAULTDEP_INPUT));
  }

  @Test
  public void configurabilityDefaults() throws Exception {
    writeHelloRules(/*includeDefaultCondition=*/true);
    writeConfigRules();
    checkRule("//java/hello:hello", "--test_arg=something_random",
        /*expected:*/ ImmutableList.of(DEFAULTDEP_INPUT),
        /*not expected:*/ ImmutableList.of(ADEP_INPUT, BDEP_INPUT));
    checkRule("//java/hello:hello", "",
        /*expected:*/ ImmutableList.of(DEFAULTDEP_INPUT),
        /*not expected:*/ ImmutableList.of(ADEP_INPUT, BDEP_INPUT));
  }

  /**
   * Duplicate label definitions are fine as long as they're in different selection branches.
   */
  @Test
  public void depsWithDuplicatesInDifferentBranches() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = ['hello.java'],",
        "    deps = select({",
        "        '//conditions:a': [':adep', ':cdep'],",
        "        '//conditions:b': [':bdep', ':cdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))",
        "",
        "java_library(",
        "    name = 'adep',",
        "    srcs = ['adep.java'])",
        "java_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.java'])",
        "java_library(",
        "    name = 'cdep',",
        "    srcs = ['cdep.java'])");
    checkRule("//java/hello:hello",  "--test_arg=a",
        /*expected:*/ ImmutableList.of(ADEP_INPUT, CDEP_INPUT),
        /*not expected:*/ ImmutableList.of(BDEP_INPUT, DEFAULTDEP_INPUT));
  }

  /**
   * Duplicate label definitions are *not* fine within the same branch.
   */
  @Test
  public void depsWithDuplicatesInSameBranch() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = ['hello.java'],",
        "    deps = select({",
        "        '//conditions:a': [':adep', ':cdep', ':adep'],",
        "        '//conditions:b': [':bdep', ':cdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))",
        "",
        "java_library(",
        "    name = 'adep',",
        "    srcs = ['adep.java'])",
        "java_library(",
        "    name = 'bdep',",
        "    srcs = ['bdep.java'])",
        "java_library(",
        "    name = 'cdep',",
        "    srcs = ['cdep.java'])");

    reporter.removeHandler(failFastHandler); // Expect errors.
    useConfiguration("--test_arg=a");
    getConfiguredTarget("//java/hello:hello");
    assertContainsEvent(
        "Label '//java/hello:adep' is duplicated in the 'deps' attribute of rule 'hello'");
  }

  /**
   * When an attribute includes multiple selects, we don't allow duplicates even across
   * selects (this saves us from having to do possibly expensive value iteration since the
   * number of values can grow exponentially with respect to the number of selects).
   */
  @Test
  public void duplicatesAcrossMultipleSelects() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = select({",
        "        '//conditions:a': ['a.java'],",
        "        '//conditions:b': ['b.java'],",
        "        })",
        "        + select({",
        "        '//conditions:c': ['c.java'],",
        "        '//conditions:d': ['a.java'],",
        "    }))");

    reporter.removeHandler(failFastHandler); // Expect errors.
    useConfiguration("--test_arg=a");
    getConfiguredTarget("//java/hello:hello");
    assertContainsEvent(
        "Label '//java/hello:a.java' is duplicated in the 'srcs' attribute of rule 'hello'");
  }

  /**
   * Even with multiple selects, duplicates are allowed within a *single* select as long as
   * they're in different branches (and thus mutually exclusive).
   */
  @Test
  public void duplicatesInDifferentBranchesMultipleSelects() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = select({",
        "        '//conditions:a': ['a.java'],",
        "        '//conditions:b': ['a.java'],",
        "        })",
        "        + select({",
        "        '//conditions:a': ['b.java'],",
        "        '//conditions:b': ['b.java'],",
        "    }))");

    useConfiguration("--test_arg=a");
    getConfiguredTarget("//java/hello:hello");
    assertNoEvents();
  }

  /**
   * With multiple selects, a single select still can't duplicate labels within the same branch.
   */
  @Test
  public void duplicatesInSameBranchMultipleSelects() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello',",
        "    srcs = select({",
        "        '//conditions:a': ['a.java', 'a.java'],",
        "        '//conditions:b': ['b.java'],",
        "        })",
        "        + select({",
        "        '//conditions:a': ['c.java'],",
        "        '//conditions:b': ['d.java'],",
        "    }))");

    reporter.removeHandler(failFastHandler); // Expect errors.
    useConfiguration("--test_arg=a");
    getConfiguredTarget("//java/hello:hello");
    assertContainsEvent(
        "Label '//java/hello:a.java' is duplicated in the 'srcs' attribute of rule 'hello'");
  }

  /**
   * Attributes of type {@link BuildType#OUTPUT} are not configurable.
   */
  @Test
  public void outputTypeNotConfigurable() throws Exception {
    writeConfigRules();
    scratch.file("foo/BUILD",
        "rule_with_output_attr(",
        "    name = 'has_an_out',",
        "    out = select({",
        "        '//conditions:a': 'a.out',",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': 'default.out'})",
        ")");

    reporter.removeHandler(failFastHandler); // Expect errors.
    getConfiguredTarget("//foo:has_an_out");
    assertContainsEvent("attribute \"out\" is not configurable");
  }

  /**
   * Attributes of type {@link BuildType#OUTPUT_LIST} are not configurable.
   */
  @Test
  public void outputListTypeNotConfigurable() throws Exception {
    writeConfigRules();
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'generator',",
        "    srcs = [],",
        "    outs = select({",
        "        '//conditions:a': ['a.out'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': ['default.out']})",
        ")");

    reporter.removeHandler(failFastHandler); // Expect errors.
    getConfiguredTarget("//foo:generator");
    assertContainsEvent("attribute \"outs\" is not configurable");
  }

  /**
   * Tests that computed defaults faithfully reflect the values of the attributes they depend on.
   */
  @Test
  public void computedDefaults() throws Exception {
    writeConfigRules();
    scratch.file("test/BUILD",
        "rule_with_computed_default(",
        "    name = 'the_rule',",
        "    string_attr = select({",
        "        '//conditions:a': 'a',",
        "        '//conditions:b': 'b',",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': 'default',",
        "    }))");

    // Configuration a:
    useConfiguration("--test_arg=a");
    ConfiguredTargetAndData binary = getConfiguredTargetAndData("//test:the_rule");
    AttributeMap attributes = getMapperFromConfiguredTargetAndTarget(binary);
    assertThat(attributes.get("$computed_attr", Type.STRING)).isEqualTo("a2");

    // configuration b:
    useConfiguration("--test_arg=b");
    binary = getConfiguredTargetAndData("//test:the_rule");
    attributes = getMapperFromConfiguredTargetAndTarget(binary);
    assertThat(attributes.get("$computed_attr", Type.STRING)).isEqualTo("b2");
  }

  @Test
  public void configKeyTypeChecking_Int() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file(
        "java/foo/BUILD",
        "java_library(",
        "    name = 'int_key',",
        "    srcs = select({123: ['a.java']})",
        ")");
    assertTargetError("//java/foo:int_key", "select: got int for dict key, want a label string");
  }

  @Test
  public void configKeyTypeChecking_Bool() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file(
        "java/foo/BUILD",
        "java_library(",
        "    name = 'bool_key',",
        "    srcs = select({True: ['a.java']})",
        ")");
    assertTargetError("//java/foo:bool_key", "select: got bool for dict key, want a label string");
  }

  @Test
  public void configKeyTypeChecking_None() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file(
        "java/foo/BUILD",
        "java_library(",
        "    name = 'none_key',",
        "    srcs = select({None: ['a.java']})",
        ")");
    assertTargetError(
        "//java/foo:none_key", "select: got NoneType for dict key, want a label string");
  }

  @Test
  public void selectWithoutConditionsMakesNoSense() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file(
        "foo/BUILD",
        "genrule(",
        "    name = 'nothing',",
        "    srcs = [],",
        "    outs = ['notmuch'],",
        "    cmd = select({})",
        ")");
    assertTargetError(
        "//foo:nothing",
        "select({}) with an empty dictionary can never resolve because it includes no conditions "
            + "to match");
  }

  /**
   * Tests that config keys must resolve to existent targets.
   */
  @Test
  public void missingConfigKey() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    // Only create one of two necessary configurability rules:
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})");
    writeHelloRules(/*includeDefaultCondition=*/true);
    getConfiguredTarget("//java/hello:hello");
    assertContainsEvent("no such target '//conditions:b'");
  }

  /**
   * Tests that config keys must resolve to config_setting targets.
   */
  @Test
  public void invalidConfigKey() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})",
        "rule_with_output_attr(",
        "    name = 'b',",
        "    out = 'b.out')");
    writeHelloRules(/*includeDefaultCondition=*/true);
    assertThat(getConfiguredTarget("//java/hello:hello")).isNull();
    assertContainsEvent("//conditions:b is not a valid select() condition for //java/hello:hello");
    assertDoesNotContainEvent("//conditions:a"); // This one is legitimate..
  }

  @Test
  public void configKeyNonexistentTarget() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'g',",
        "    outs = ['g.out'],",
        "    cmd = select({':fake': ''})",
        ")");
    assertThat(getConfiguredTarget("//foo:g")).isNull();
    assertContainsEvent("//foo:fake is not a valid select() condition for //foo:g");
  }

  @Test
  public void configKeyNonexistentTarget_otherPackage() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file(
        "conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'a'})");
    scratch.file("bar/BUILD");
    scratch.file(
        "foo/BUILD",
        "genrule(",
        "    name = 'g',",
        "    outs = ['g.out'],",
        // With an invalid target and a real target, validate skyframe error handling.
        // See http://b/162021059 for details.
        "    cmd = select({'//bar:fake': '', '//conditions:a': ''})",
        ")");
    assertThat(getConfiguredTarget("//foo:g")).isNull();
    assertContainsEvent(
        "While resolving configuration keys for //foo:g: no such target '//bar:fake'");
  }

  /**
   * Tests config keys with multiple requirements.
   */
  @Test
  public void multiConditionConfigKeys() throws Exception {
    writeHelloRules(/*includeDefaultCondition=*/true);
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {",
        "        'test_arg': 'a',",
        "        'compilation_mode': 'dbg'",
        "    })",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");
    checkRule("//java/hello:hello", "--test_arg=a",
        /*expected:*/ ImmutableList.of(DEFAULTDEP_INPUT),
        /*not expected:*/ ImmutableList.of(ADEP_INPUT, BDEP_INPUT));
    checkRule("//java/hello:hello", ImmutableList.of("--test_arg=a", "--compilation_mode=dbg"),
        /*expected:*/ ImmutableList.of(ADEP_INPUT),
        /*not expected:*/ ImmutableList.of(BDEP_INPUT, DEFAULTDEP_INPUT));
  }

  /**
   * Tests that changing a config_setting invalidates the rule that uses it.
   */
  @Test
  public void configKeyInvalidation() throws Exception {
    writeHelloRules(/*includeDefaultCondition=*/true);
    writeConfigRules();

    // Iteration 1: --test_args=a should apply //conditions:a.
    useConfiguration("--test_arg=a");
    checkRule("//java/hello:hello", "--test_arg=a",
        /*expected:*/ ImmutableList.of(ADEP_INPUT),
        /*not expected:*/ ImmutableList.of(BDEP_INPUT, DEFAULTDEP_INPUT));

    // Rewrite the condition for //conditions:a.
    scratch.overwriteFile("conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'test_arg': 'c'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'test_arg': 'b'})");

    // Iteration 2: same exact analysis should now apply the default condition.
    invalidatePackages();
    checkRule("//java/hello:hello", "--test_arg=a",
        /*expected:*/ ImmutableList.of(DEFAULTDEP_INPUT),
        /*not expected:*/ ImmutableList.of(ADEP_INPUT, BDEP_INPUT));
  }

  /**
   * Tests that multiple matches are not allowed for conditions where one is not a specialization
   * of the other.
   */
  @Test
  public void multipleMatches() throws Exception {
    reporter.removeHandler(failFastHandler); // Expect errors.
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'dup1',",
        "    values = {'compilation_mode': 'opt'})",
        "config_setting(",
        "    name = 'dup2',",
        "    values = {'define': 'foo=bar'})");
    scratch.file("a/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    cmd = '',",
        "    outs = ['gen.out'],",
        "    srcs = select({",
        "        '//conditions:dup1': ['a.in'],",
        "        '//conditions:dup2': ['b.in'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':default.in'],",
        "    }))");
    useConfiguration("-c", "opt", "--define", "foo=bar");
    assertThat(getConfiguredTarget("//a:gen")).isNull();
    assertContainsEvent(
        "Illegal ambiguous match on configurable attribute \"srcs\" in //a:gen:\n"
            + "//conditions:dup1\n"
            + "//conditions:dup2\n"
            + "Multiple matches are not allowed unless one is unambiguously more specialized.");
  }

  /**
   * Tests that when multiple conditions match and for every matching pair, one is
   * a specialization of the other, the most specialized match is chosen.
   */
  @Test
  public void multipleMatchesConditionAndSubcondition() throws Exception {
    scratch.file("conditions/BUILD",
        "config_setting(",
        "    name = 'generic',",
        "    values = {'compilation_mode': 'opt'})",
        "config_setting(",
        "    name = 'precise',",
        "    values = {'compilation_mode': 'opt', 'define': 'foo=bar'})",
        "config_setting(",
        "    name = 'most_precise',",
        "    values = {'compilation_mode': 'opt', 'define': 'foo=bar', 'test_arg': 'baz'})");
        scratch.file("java/a/BUILD",
            "java_binary(",
            "    name = 'binary',",
            "    srcs = ['binary.java'],",
            "    deps = select({",
            "        '//conditions:generic': [':generic'],",
            "        '//conditions:precise': [':precise'],",
            "        '//conditions:most_precise': [':most_precise'],",
            "    }))",
            "java_library(",
            "    name = 'generic',",
            "    srcs = ['generic.java'])",
            "java_library(",
            "    name = 'precise',",
            "    srcs = ['precise.java'])",
            "java_library(",
            "    name = 'most_precise',",
            "    srcs = ['most_precise.java'])");
    checkRule("//java/a:binary",
        ImmutableList.of("-c", "opt", "--define", "foo=bar", "--test_arg", "baz"),
        /*expected:*/ ImmutableList.of("bin java/a/libmost_precise.jar"),
        /*not expected:*/ ImmutableList.of(
            "bin java/a/libgeneric.jar",
            "bin java/a/libprecise.jar"));
  }

  /**
   * Tests that when multiple conditions match but one condition is more specialized than the
   * others, it is chosen and there is no error.
   */
  @Test
  public void multipleMatchesUnambiguous() throws Exception {
    scratch.file(
        "conditions/BUILD",
        "config_setting(",
        "    name = 'a',",
        "    values = {'define': 'a=1'})",
        "config_setting(",
        "    name = 'b',",
        "    values = {'compilation_mode': 'opt'})",
        "config_setting(",
        "    name = 'c',",
        "    values = {'test_arg': 'baz'})",
        "config_setting(",
        "    name = 'b_a_c',", // Named to come alphabetically after a and b but before c.
        "    values = {'define': 'a=1', 'test_arg': 'baz', 'compilation_mode': 'opt'})");
    scratch.file("java/a/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = ['binary.java'],",
        "    deps = select({",
        "        '//conditions:a': [':a'],",
        "        '//conditions:b': [':b'],",
        "        '//conditions:c': [':c'],",
        "        '//conditions:b_a_c': [':b_a_c'],",
        "    }))",
        "java_library(",
        "    name = 'a',",
        "    srcs = ['a.java'])",
        "java_library(",
        "    name = 'b',",
        "    srcs = ['b.java'])",
        "java_library(",
        "    name = 'c',",
        "    srcs = ['c.java'])",
        "java_library(",
        "    name = 'b_a_c',",
        "    srcs = ['b_a_c.java'])");
    checkRule(
        "//java/a:binary",
        ImmutableList.of("--define", "a=1", "--compilation_mode", "opt", "--test_arg", "baz"),
        /*expected:*/ ImmutableList.of("bin java/a/libb_a_c.jar"),
        /*not expected:*/ ImmutableList.of(
            "bin java/a/liba.jar", "bin java/a/libb.jar", "bin java/a/libc.jar"));
  }

  /** Tests that default conditions are only required when no main condition matches. */
  @Test
  public void noDefaultCondition() throws Exception {
    writeHelloRules(/*includeDefaultCondition=*/false);
    writeConfigRules();

    // An explicit configuration matches: all is well.
    checkRule("//java/hello:hello", "--test_arg=a",
        /*expected:*/ ImmutableList.of(ADEP_INPUT),
        /*not expected:*/ ImmutableList.of(BDEP_INPUT, DEFAULTDEP_INPUT));

    // Nothing matches: expect an error.
    reporter.removeHandler(failFastHandler);
    useConfiguration("");
    assertThat(getConfiguredTarget("//java/hello:hello")).isNull();
    assertContainsEvent("Configurable attribute \"deps\" doesn't match this configuration");
  }

  @Test
  public void noMatchCustomErrorMessage() throws Exception {
    writeConfigRules();
    scratch.file("java/hello/BUILD",
        "java_binary(",
        "    name = 'hello_default_no_match_error',",
        "    srcs = select({",
        "        '//conditions:a': ['not_chosen.java'],",
        "    }))",
        "java_binary(",
        "    name = 'hello_custom_no_match_error',",
        "    srcs = select({",
        "        '//conditions:a': ['not_chosen.java'],",
        "    },",
        "    no_match_error = 'You always have to choose condition a!'",
        "))");

    reporter.removeHandler(failFastHandler);
    AnalysisFailureRecorder analysisFailureRecorder = new AnalysisFailureRecorder();
    eventBus.register(analysisFailureRecorder);
    useConfiguration("");

    assertThat(getConfiguredTarget("//java/hello:hello_default_no_match_error")).isNull();
    String commonPrefix = "Configurable attribute \"srcs\" doesn't match this configuration";
    assertContainsEvent(commonPrefix + " (would a default condition help?).\nConditions checked:");
    // Verify a Root Cause is reported when a target cannot be configured due to no matching config.
    assertThat(analysisFailureRecorder.causes).hasSize(1);
    AnalysisRootCauseEvent rootCause = analysisFailureRecorder.causes.get(0);
    assertThat(rootCause.getLabel())
        .isEqualTo(
            Label.parseAbsolute("//java/hello:hello_default_no_match_error", ImmutableMap.of()));

    eventBus.unregister(analysisFailureRecorder);
    analysisFailureRecorder = new AnalysisFailureRecorder();
    eventBus.register(analysisFailureRecorder);
    eventCollector.clear();

    assertThat(getConfiguredTarget("//java/hello:hello_custom_no_match_error")).isNull();
    assertContainsEvent(commonPrefix + ": You always have to choose condition a!");
    // Verify a Root Cause is reported when a target cannot be configured due to no matching config.
    assertThat(analysisFailureRecorder.causes).hasSize(1);
    rootCause = analysisFailureRecorder.causes.get(0);
    assertThat(rootCause.getLabel())
        .isEqualTo(
            Label.parseAbsolute("//java/hello:hello_custom_no_match_error", ImmutableMap.of()));
  }

  @Test
  public void nativeTypeConcatenatedWithSelect() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = ['binary.java'],",
        "    deps = [':always'] + select({",
        "        '//conditions:a': [':a'],",
        "        '//conditions:b': [':b'],",
        "    })",
        ")",
        "java_library(",
        "    name = 'always',",
        "    srcs = ['always.java'])",
        "java_library(",
        "    name = 'a',",
        "    srcs = ['a.java'])",
        "java_library(",
        "    name = 'b',",
        "    srcs = ['b.java'])");

    checkRule("//java/foo:binary", "--test_arg=b",
        /*expected:*/ ImmutableList.of(
            "bin java/foo/libalways.jar",
            "bin java/foo/libb.jar"),
        /*not expected:*/ ImmutableList.of(
            "bin java/foo/liba.jar"));
  }

  @Test
  public void selectConcatenatedWithNativeType() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = ['binary.java'],",
        "    deps = select({",
        "        '//conditions:a': [':a'],",
        "        '//conditions:b': [':b'],",
        "    }) + [':always'])",
        "java_library(",
        "    name = 'always',",
        "    srcs = ['always.java'])",
        "java_library(",
        "    name = 'a',",
        "    srcs = ['a.java'])",
        "java_library(",
        "    name = 'b',",
        "    srcs = ['b.java'])");

    checkRule("//java/foo:binary", "--test_arg=b",
        /*expected:*/ ImmutableList.of(
            "bin java/foo/libalways.jar",
            "bin java/foo/libb.jar"),
        /*not expected:*/ ImmutableList.of(
            "bin java/foo/liba.jar"));
  }

  @Test
  public void selectConcatenatedWithSelect() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = ['binary.java'],",
        "    deps = select({",
        "        '//conditions:a': [':a'],",
        "        '//conditions:b': [':b'],",
        "    }) + select({",
        "        '//conditions:a': [':a2'],",
        "        '//conditions:b': [':b2'],",
        "    })",
        ")",
        "java_library(",
        "    name = 'a',",
        "    srcs = ['a.java'])",
        "java_library(",
        "    name = 'b',",
        "    srcs = ['b.java'])",
        "java_library(",
        "    name = 'a2',",
        "    srcs = ['a2.java'])",
        "java_library(",
        "    name = 'b2',",
        "    srcs = ['b2.java'])");

    checkRule("//java/foo:binary", "--test_arg=b",
        /*expected:*/ ImmutableList.of(
            "bin java/foo/libb.jar",
            "bin java/foo/libb2.jar"),
        /*not expected:*/ ImmutableList.of(
            "bin java/foo/liba.jar",
            "bin java/foo/liba2.jar"));
  }

  @Test
  public void selectConcatenatedWithNonSupportingType() throws Exception {
    writeConfigRules();
    scratch.file("foo/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= 0 + select({",
        "        '//conditions:a': 0,",
        "        '//conditions:b': 1,",
        "    }))");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//foo:binary")).isNull();
    assertContainsEvent("type 'boolean' doesn't support select concatenation");
  }

  @Test
  public void concatenationWithDifferentTypes() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = select({",
        "        '//conditions:a': ['a.java'],",
        "        '//conditions:b': ['b.java'],",
        "    }) + 'always.java'",
        ")");

    reporter.removeHandler(failFastHandler);
    assertThrows(NoSuchTargetException.class, () -> getTarget("//java/foo:binary"));
    assertContainsEvent("'+' operator applied to incompatible types");
  }

  @Test
  public void selectsWithGlobs() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/globbed/ceecee.java");
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = glob(['globbed/*.java']) + select({",
        "        '//conditions:a': ['a.java'],",
        "        '//conditions:b': ['b.java'],",
        "    }))");

    useConfiguration("--test_arg=b");
    ConfiguredTarget binary = getConfiguredTarget("//java/foo:binary");
    assertThat(binary).isNotNull();
    Set<String> sources = artifactsToStrings(getPrerequisiteArtifacts(binary, "srcs"));
    assertThat(sources).contains("src java/foo/b.java");
    assertThat(sources).contains("src java/foo/globbed/ceecee.java");
    assertThat(sources).doesNotContain("src java/foo/a.java");
  }

  @Test
  public void selectsWithGlobsWrongType() throws Exception {
    writeConfigRules();
    scratch.file("foo/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['gen.out'],",
        "    cmd = 'echo' + select({",
        "        '//conditions:a': 'a',",
        "        '//conditions:b': 'b',",
        "    }) + glob(['globbed.java']))");

    reporter.removeHandler(failFastHandler);
    assertThrows(NoSuchTargetException.class, () -> getTarget("//foo:binary"));
    assertContainsEvent("'+' operator applied to incompatible types");
  }

  @Test
  public void globsInSelect() throws Exception {
    writeConfigRules();
    scratch.file("java/foo/globbed/ceecee.java");
    scratch.file("java/foo/BUILD",
        "java_binary(",
        "    name = 'binary',",
        "    srcs = ['binary.java'] + select({",
        "        '//conditions:a': glob(['globbed/*.java']),",
        "        '//conditions:b': ['b.java'],",
        "    }))");

    useConfiguration("--test_arg=a");
    ConfiguredTarget binary = getConfiguredTarget("//java/foo:binary");
    assertThat(binary).isNotNull();
    Set<String> sources = artifactsToStrings(getPrerequisiteArtifacts(binary, "srcs"));
    assertThat(sources).contains("src java/foo/binary.java");
    assertThat(sources).contains("src java/foo/globbed/ceecee.java");
    assertThat(sources).doesNotContain("src java/foo/b.java");
  }

  @Test
  public void selectAcceptedInAttributeWithAllowedValues() throws Exception {
    scratch.file("foo/BUILD",
        "rule_with_allowed_values(",
        "    name = 'rule',",
        "    one_two = select({",
        "        '//conditions:default': 'one',",
        "    }))");
    assertThat(getConfiguredTarget("//foo:rule")).isNotNull();
  }

  @Test
  public void selectWithNonAllowedValueCausesError() throws Exception {
    scratch.file("foo/BUILD",
        "rule_with_allowed_values(",
        "    name = 'rule',",
        "    one_two = select({",
        "        '//conditions:default': 'TOTALLY_ILLEGAL_VALUE',",
        "    }))");
    reporter.removeHandler(failFastHandler); // Expect errors.
    getConfiguredTarget("//foo:rule");
    assertContainsEvent(
        "invalid value in 'one_two' attribute: "
        + "has to be one of 'one' or 'two' instead of 'TOTALLY_ILLEGAL_VALUE'");
  }

  @Test
  public void selectWithMultipleNonAllowedValuesCausesMultipleErrors() throws Exception {
    scratch.file("foo/BUILD",
        "rule_with_allowed_values(",
        "    name = 'rule',",
        "    one_two = select({",
        "        '//conditions:a': 'TOTALLY_ILLEGAL_VALUE',",
        "        '//conditions:default': 'DIFFERENT_BUT_STILL_ILLEGAL',",
        "    }))");
    reporter.removeHandler(failFastHandler); // Expect errors.
    getConfiguredTarget("//foo:rule");
    assertContainsEvent(
        "invalid value in 'one_two' attribute: "
        + "has to be one of 'one' or 'two' instead of 'TOTALLY_ILLEGAL_VALUE'");
    assertContainsEvent(
        "invalid value in 'one_two' attribute: "
        + "has to be one of 'one' or 'two' instead of 'DIFFERENT_BUT_STILL_ILLEGAL'");
  }

  @Test
  public void selectConcatenationWithAllowedValues() throws Exception {
    scratch.file("foo/BUILD",
           "rule_with_allowed_values(",
           "    name = 'rule',",
           "    one_two = 'on' + select({",
           "        '//conditions:default': 'e',",
           "    }))");
    assertThat(getConfiguredTarget("//foo:rule")).isNotNull();
  }

  @Test
  public void selectConcatenationWithNonAllowedValues() throws Exception {
    scratch.file("foo/BUILD",
           "rule_with_allowed_values(",
           "    name = 'rule',",
           "    one_two = 'on' + select({",
           "        '//conditions:default': 'o',",
           "    }))");
    reporter.removeHandler(failFastHandler); // Expect errors.
    getConfiguredTarget("//foo:binary");
    assertContainsEvent(
        "invalid value in 'one_two' attribute: "
        + "has to be one of 'one' or 'two' instead of 'ono'");
  }

  @Test
  public void computedDefaultAttributesCanReferenceConfigurableAttributes() throws Exception {
    scratch.file(
        "test/selector_rules.bzl",
        "def _impl(ctx):",
        "  ctx.actions.write(",
        "      output=ctx.outputs.out_file,",
        "      content=ctx.attr.string_value,",
        "  )",
        "  return []",
        "",
        "def _derived_value(string_value):",
        "  return Label(\"//test:%s\" % string_value)",
        "",
        "selector_rule = rule(",
        "  attrs = {",
        "      \"string_value\": attr.string(default = \"\"),",
        "      \"out_file\": attr.output(),",
        "      \"_derived\": attr.label(default = _derived_value),",
        "  },",
        "implementation = _impl,",
        ")");
    scratch.file("test/BUILD",
        "genrule(name = \"foo\", srcs = [], outs = [\"foo.out\"], cmd = \"\")");
    scratch.file(
        "foo/BUILD",
        "load('//test:selector_rules.bzl', \"selector_rule\")",
        "selector_rule(",
        "    name = \"rule\",",
        "    out_file = \"rule.out\",",
        "    string_value = select({\"//conditions:default\": \"foo\"}),",
        ")");
    getConfiguredTarget("//foo:rule");
    assertNoEvents();
  }

  @Test
  public void selectableDefaultValueWithTypeDefault() throws Exception {
    writeConfigRules();
    scratch.file("srctest/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    cmd = '',",
        "    outs = ['gen.out'],",
        "    srcs = select({",
        "        '//conditions:a': None,",
        "    }))");

    useConfiguration("--test_arg=a");
    ConfiguredTargetAndData ctad = getConfiguredTargetAndData("//srctest:gen");
    AttributeMap attributes = getMapperFromConfiguredTargetAndTarget(ctad);
    assertThat(attributes.get("srcs", LABEL_LIST)).isEmpty();
  }

  @Test
  public void selectableDefaultValueWithRuleDefault() throws Exception {
    writeConfigRules();
    scratch.file("foo/BUILD",
        "rule_with_label_default(",
        "    name = 'rule',",
        "    dep = select({",
        "        '//conditions:a': None,",
        "    }))",
        "rule_with_boolean_attr(",
        "    name = 'default',",
        "    boolean_attr = 1)");

    useConfiguration("--test_arg=a");
    ConfiguredTargetAndData ctad = getConfiguredTargetAndData("//foo:rule");
    AttributeMap attributes = getMapperFromConfiguredTargetAndTarget(ctad);
    assertThat(attributes.get("dep", BuildType.LABEL))
        .isEqualTo(Label.parseAbsolute("//foo:default", ImmutableMap.of()));
  }

  @Test
  public void noneValuesWithMultipleSelectsMixedValues() throws Exception {
    writeConfigRules();
    scratch.file("a/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = '',",
        "    message = select({",
        "        '//conditions:a': 'defined message 1',",
        "        '//conditions:b': None,",
        "    }) + select({",
        "        '//conditions:a': None,",
        "        '//conditions:b': 'defined message 2',",
        "    }),",
        ")");

    reporter.removeHandler(failFastHandler);
    useConfiguration("--define", "mode=a");
    assertThat(getConfiguredTarget("//a:gen")).isNull();
    assertContainsEvent(
        "'+' operator applied to incompatible types (select of string, select of NoneType)");
  }

  @Test
  public void emptySelectCannotBeConcatenated() throws Exception {
    scratch.file("a/BUILD",
        "genrule(",
        "    name = 'gen',",
        "    srcs = [],",
        "    outs = ['out'],",
        "    cmd = select({}) + ' always include'",
        ")");

    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:gen")).isNull();
    assertContainsEvent(
        "select({}) with an empty dictionary can never resolve because it includes no conditions "
            + "to match");
  }

  @Test
  public void selectOnConstraints() throws Exception {
    // create some useful constraints and platforms.
    scratch.file(
        "conditions/BUILD",
        "constraint_setting(name = 'fruit')",
        "constraint_value(name = 'apple', constraint_setting = 'fruit')",
        "constraint_value(name = 'banana', constraint_setting = 'fruit')",
        "platform(",
        "    name = 'apple_platform',",
        "    constraint_values = [':apple'],",
        ")",
        "platform(",
        "    name = 'banana_platform',",
        "    constraint_values = [':banana'],",
        ")",
        "config_setting(",
        "    name = 'a',",
        "    constraint_values = [':apple']",
        ")",
        "config_setting(",
        "    name = 'b',",
        "    constraint_values = [':banana']",
        ")");
    scratch.file(
        "check/BUILD",
        "filegroup(name = 'adep', srcs = ['afile'])",
        "filegroup(name = 'bdep', srcs = ['bfile'])",
        "filegroup(name = 'defaultdep', srcs = ['defaultfile'])",
        "filegroup(name = 'hello',",
        "    srcs = select({",
        "        '//conditions:a': [':adep'],",
        "        '//conditions:b': [':bdep'],",
        "        '" + BuildType.Selector.DEFAULT_CONDITION_KEY + "': [':defaultdep'],",
        "    }))");
    checkRule(
        "//check:hello",
        "srcs",
        ImmutableList.of("--experimental_platforms=//conditions:apple_platform"),
        /*expected:*/ ImmutableList.of("src check/afile"),
        /*not expected:*/ ImmutableList.of("src check/bfile", "src check/defaultfile"));
  }

  @Test
  public void selectDirectlyOnConstraints() throws Exception {
    // Tests select()ing directly on a constraint_value (with no intermediate config_setting).
    scratch.file(
        "conditions/BUILD",
        "constraint_setting(name = 'fruit')",
        "constraint_value(name = 'apple', constraint_setting = 'fruit')",
        "constraint_value(name = 'banana', constraint_setting = 'fruit')",
        "platform(",
        "    name = 'apple_platform',",
        "    constraint_values = [':apple'],",
        ")",
        "platform(",
        "    name = 'banana_platform',",
        "    constraint_values = [':banana'],",
        ")");
    scratch.file(
        "check/defs.bzl",
        "def _impl(ctx):",
        "  pass",
        "simple_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {'srcs': attr.label_list(allow_files = True)}",
        ")");
    scratch.file(
        "check/BUILD",
        "load('//check:defs.bzl', 'simple_rule')",
        "filegroup(name = 'adep', srcs = ['afile'])",
        "filegroup(name = 'bdep', srcs = ['bfile'])",
        "simple_rule(name = 'hello',",
        "    srcs = select({",
        "        '//conditions:apple': [':adep'],",
        "        '//conditions:banana': [':bdep'],",
        "    }))");
    checkRule(
        "//check:hello",
        "srcs",
        ImmutableList.of("--platforms=//conditions:apple_platform"),
        /*expected:*/ ImmutableList.of("src check/afile"),
        /*not expected:*/ ImmutableList.of("src check/bfile", "src check/defaultfile"));
    checkRule(
        "//check:hello",
        "srcs",
        ImmutableList.of("--platforms=//conditions:banana_platform"),
        /*expected:*/ ImmutableList.of("src check/bfile"),
        /*not expected:*/ ImmutableList.of("src check/afile", "src check/defaultfile"));
  }

  @Test
  public void nonToolchainResolvingTargetsCantSelectDirectlyOnConstraints() throws Exception {
    // Tests select()ing directly on a constraint_value (with no intermediate config_setting).
    scratch.file(
        "conditions/BUILD",
        "constraint_setting(name = 'fruit')",
        "constraint_value(name = 'apple', constraint_setting = 'fruit')",
        "platform(",
        "    name = 'apple_platform',",
        "    constraint_values = [':apple'],",
        ")");
    scratch.file(
        "check/BUILD",
        "filegroup(name = 'adep', srcs = ['afile'])",
        "rule_with_no_platform(name = 'hello',",
        "    deps = select({",
        "        '//conditions:apple': [':adep'],",
        "    })",
        ")");
    reporter.removeHandler(failFastHandler);
    useConfiguration("--platforms=//conditions:apple_platform");
    assertThat(getConfiguredTarget("//check:hello")).isNull();
    assertContainsEvent("//conditions:apple is not a valid select() condition for //check:hello");
  }

  @Test
  public void multipleMatchErrorWhenAliasResolvesToSameSetting() throws Exception {
    scratch.file(
        "a/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' })",
        "alias(",
        "    name = 'alias_to_foo',",
        "    actual = ':foo')",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        ':foo': 0,",
        "        'alias_to_foo': 1,",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNull();
    assertContainsEvent(
        "Configurable attribute \"boolean_attr\" doesn't match this configuration (would a default "
            + "condition help?).\n"
            + "Conditions checked:\n"
            + " //a:foo\n"
            + " //a:alias_to_foo");
  }

  @Test
  public void defaultVisibilityConfigSetting_noVisibilityEnforcement() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=false");
    scratch.file("c/BUILD", "config_setting(name = 'foo', define_values = { 'foo': '1' })");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void privateVisibilityConfigSetting_noVisibilityEnforcement() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=false");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:private']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void publicVisibilityConfigSetting__noVisibilityEnforcement() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=false");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:public']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }
  @Test
  public void defaultVisibilityConfigSetting_defaultIsPublic() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=false");
    scratch.file("c/BUILD", "config_setting(name = 'foo', define_values = { 'foo': '1' })");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }

  @Test
  public void privateVisibilityConfigSetting_defaultIsPublic() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=false");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:private']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNull();
    assertContainsEvent("'//c:foo' is not visible from target '//a:binary'");
  }

  @Test
  public void publicVisibilityConfigSetting_defaultIsPublic() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=false");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:public']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }
  @Test
  public void defaultVisibilityConfigSetting_defaultIsPrivate() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=true");
    scratch.file("c/BUILD", "config_setting(name = 'foo', define_values = { 'foo': '1' })");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNull();
    assertContainsEvent("'//c:foo' is not visible from target '//a:binary'");
  }

  @Test
  public void privateVisibilityConfigSetting_defaultIsPrivate() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=true");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:private']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNull();
    assertContainsEvent("'//c:foo' is not visible from target '//a:binary'");
  }

  @Test
  public void publicVisibilityConfigSetting_defaultIsPrivate() throws Exception {
    // Production builds default to private visibility, but BuildViewTestCase defaults to public.
    setPackageOptions("--default_visibility=private",
        "--incompatible_enforce_config_setting_visibility=true",
        "--incompatible_config_setting_private_default_visibility=true");
    scratch.file(
        "c/BUILD",
        "config_setting(",
        "    name = 'foo',",
        "    define_values = { 'foo': '1' },",
        "    visibility = ['//visibility:public']",
        ")");
    scratch.file(
        "a/BUILD",
        "rule_with_boolean_attr(",
        "    name = 'binary',",
        "    boolean_attr= select({",
        "        '//c:foo': 0,",
        "        '//conditions:default': 1",
        "    }))");
    reporter.removeHandler(failFastHandler);
    assertThat(getConfiguredTarget("//a:binary")).isNotNull();
    assertNoEvents();
  }
}
