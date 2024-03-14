// Copyright 2018 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.query2.cquery;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.BuildOptionsView;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.util.DummyTestFragment.DummyTestOptions;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.packages.LabelPrinter;
import com.google.devtools.build.lib.query2.common.CqueryNode;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.ConfigurableQuery;
import com.google.devtools.build.lib.server.FailureDetails.Query;
import com.google.devtools.build.lib.server.FailureDetails.Query.Code;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Iterator;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ConfiguredTargetQueryEnvironment}.
 *
 * <p>This tests core cquery behavior (behavior that doesn't depend on <code>--output</code>).
 * Output format-specific behavior is covered in dedicated test classes.
 */
@RunWith(JUnit4.class)
public class ConfiguredTargetQuerySemanticsTest extends ConfiguredTargetQueryTest {
  @Test
  public void testConfigurationRespected() throws Exception {
    writeBuildFilesWithConfigurableAttributesUnconditionally();
    assertThat(eval("deps(//configurable:main) ^ //configurable:adep")).isEmpty();
    assertThat(eval("deps(//configurable:main) ^ //configurable:defaultdep")).hasSize(1);
  }

  private void setUpLabelsFunctionTests() throws Exception {
    MockRule ruleWithTransitions =
        () ->
            MockRule.define(
                "rule_with_transitions",
                attr("patch_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(TransitionFactories.of(new FooPatchTransition("SET BY PATCH"))),
                attr("string_dep", STRING),
                attr("split_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(
                        TransitionFactories.of(
                            new FooSplitTransition("SET BY SPLIT 1", "SET BY SPLIT 2"))),
                attr("patch_dep_list", LABEL_LIST)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(TransitionFactories.of(new FooPatchTransition("SET BY PATCH 2"))));
    MockRule noAttributeRule = () -> MockRule.define("no_attribute_rule");

    helper.useRuleClassProvider(
        setRuleClassProviders(ruleWithTransitions, noAttributeRule).build());

    writeFile(
        "test/BUILD",
        "rule_with_transitions(name = 'my_rule',",
        "  patch_dep = ':dep-1',",
        "  split_dep = ':dep-2',",
        "  string_dep = 'some string',",
        "  patch_dep_list = [':dep-3', ':dep-4']",
        ")",
        "no_attribute_rule(name = 'dep-1')",
        "no_attribute_rule(name = 'dep-2')",
        "no_attribute_rule(name = 'dep-3')",
        "no_attribute_rule(name = 'dep-4')");

    helper.setUniverseScope("//test:*");
  }

  @Test
  public void testLabelFunction_getsCorrectConfigurations() throws Exception {
    setUpLabelsFunctionTests();

    // (Test that you can use the labels function without an error (b/112593112)).
    // Note - 'labels' as a command for cquery is a slight misnomer since it always prints
    // labels AND configurations. But still a helpful function so oh well.
    assertThat(Iterables.getOnlyElement(eval("labels('patch_dep', //test:my_rule)"))).isNotNull();
  }

  @Test
  public void testLabelFunction_getCorrectlyConfiguredDeps() throws Exception {
    setUpLabelsFunctionTests();

    // Test that this retrieves the correctly configured version(s) of the dep(s).
    CqueryNode patchDep = Iterables.getOnlyElement(eval("labels('patch_dep', //test:my_rule)"));
    CqueryNode myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();
    assertThat(patchDep.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
  }

  @Test
  public void testLabelsFunction_splitTransitionAttribute() throws Exception {
    setUpLabelsFunctionTests();

    CqueryNode myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();

    Set<CqueryNode> splitDeps = eval("labels('split_dep', //test:my_rule)");
    assertThat(splitDeps).hasSize(2);
    for (CqueryNode kct : splitDeps) {
      assertThat(kct.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
    }
  }

  @Test
  public void testLabelsFunction_labelListAttribute() throws Exception {
    setUpLabelsFunctionTests();

    CqueryNode myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();

    // Test that this works for label_lists as well.
    Set<CqueryNode> deps = eval("labels('patch_dep_list', //test:my_rule)");
    assertThat(deps).hasSize(2);
    for (CqueryNode kct : deps) {
      assertThat(kct.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
    }
  }

  @Test
  public void testLabelsFunction_errorsOnBadAttribute() throws Exception {
    setUpLabelsFunctionTests();

    // Test that the proper error is thrown when requesting an attribute that doesn't exist.
    EvalThrowsResult evalThrowsResult = evalThrows("labels('fake_attr', //test:my_rule)", true);
    assertConfigurableQueryCode(
        evalThrowsResult.getFailureDetail(), ConfigurableQuery.Code.ATTRIBUTE_MISSING);
    assertThat(evalThrowsResult.getMessage())
        .isEqualTo(
            "in 'fake_attr' of rule //test:my_rule: configured target of type"
                + " rule_with_transitions does not have attribute 'fake_attr'");
  }

  @Test
  public void testLabelsFunction_nonLabelAttribute() throws Exception {
    setUpLabelsFunctionTests();
    assertThat(eval("labels('string_dep', //test:my_rule)")).isEmpty();
  }

  /**
   * Regression test for b/162431514. the {@code labels} query operator uses {@link
   * ConfiguredTargetAccessor#getPrerequisites} which is the actual logic being tested here.
   */
  @Test
  public void testGetPrerequisitesFromAliasReturnsActualPrerequisites() throws Exception {
    MockRule ruleWithDep =
        () ->
            MockRule.define(
                "rule_with_dep", attr("dep", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE));

    helper.useRuleClassProvider(setRuleClassProviders(ruleWithDep).build());
    writeFile(
        "test/BUILD",
        "alias(name = 'alias', actual = ':actual')",
        "rule_with_dep(name = 'actual', dep = ':dep')",
        "rule_with_dep(name = 'dep')");

    CqueryNode dep = Iterables.getOnlyElement(eval("labels('dep', '//test:alias')"));
    assertThat(dep.getLabel()).isEqualTo(Label.parseCanonicalUnchecked("//test:dep"));
  }

  @Test
  public void testAlias_filtering() throws Exception {
    MockRule ruleWithExecDep =
        () ->
            MockRule.define(
                "rule_with_exec_dep",
                attr("exec_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(ExecutionTransitionFactory.createFactory()),
                attr("$impl_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .value(Label.parseCanonicalUnchecked("//test:other")));
    MockRule simpleRule = () -> MockRule.define("simple_rule");

    helper.useRuleClassProvider(setRuleClassProviders(ruleWithExecDep, simpleRule).build());
    writeFile(
        "test/BUILD",
        "alias(name = 'other_my_rule', actual = ':my_rule')",
        "rule_with_exec_dep(name = 'my_rule', exec_dep = ':exec_dep')",
        "alias(name = 'other_exec_dep', actual = ':exec_dep')",
        "simple_rule(name='exec_dep')",
        "alias(name = 'other_impl_dep', actual = 'impl_dep')",
        "simple_rule(name='impl_dep')");

    CqueryNode other = Iterables.getOnlyElement(eval("//test:other_my_rule"));
    CqueryNode myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    // Note: {@link ConfiguredTarget#getLabel} returns the label of the "actual" value not the
    // label of the alias, so we need to check the underlying label.
    assertThat(other.getLabel()).isEqualTo(myRule.getLabel());

    // Regression test for b/73496081 in which alias-ed configured targets were skipping filtering.
    helper.setQuerySettings(Setting.ONLY_TARGET_DEPS, Setting.NO_IMPLICIT_DEPS);
    assertThat(
            evalToListOfStrings(
                "deps(//test:other_my_rule)-//test:other_my_rule"
                    + getDependencyCorrectionWithGen()))
        .isEqualTo(evalToListOfStrings("//test:my_rule"));
  }

  @Test
  public void testTopLevelTransition() throws Exception {
    MockRule ruleClassTransition =
        () ->
            MockRule.define(
                "rule_class_transition",
                (builder, env) ->
                    builder.cfg(unused -> new FooPatchTransition("SET BY PATCH")).build());

    helper.useRuleClassProvider(setRuleClassProviders(ruleClassTransition).build());
    helper.setUniverseScope("//test:rule_class");

    writeFile("test/BUILD", "rule_class_transition(name='rule_class')");

    Set<CqueryNode> ruleClass = eval("//test:rule_class");
    DummyTestOptions testOptions =
        getConfiguration(Iterables.getOnlyElement(ruleClass))
            .getOptions()
            .get(DummyTestOptions.class);
    assertThat(testOptions.foo).isEqualTo("SET BY PATCH");
  }

  private void createConfigRulesAndBuild() throws Exception {
    MockRule ruleWithTransitions =
        () ->
            MockRule.define(
                "my_rule",
                attr("target", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
                attr("exec", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(ExecutionTransitionFactory.createFactory()),
                attr("deps", BuildType.LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE));
    MockRule simpleRule =
        () ->
            MockRule.define(
                "simple_rule", attr("dep", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE));
    helper.useRuleClassProvider(setRuleClassProviders(ruleWithTransitions, simpleRule).build());

    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  target = ':target_dep',",
        "  exec = ':exec_dep',",
        "  deps = [':dep'],",
        ")",
        "simple_rule(name = 'target_dep', dep=':dep')",
        "simple_rule(name = 'exec_dep', dep=':dep')",
        "simple_rule(name = 'dep')");
  }

  private void createConfigTransitioningRuleClass() throws Exception {
    overwriteFile(
        "tools/allowlists/function_transition_allowlist/BUILD",
        "package_group(",
        "    name = 'function_transition_allowlist',",
        "    packages = [",
        "        '//test/...',",
        "    ],",
        ")");
    writeFile(
        "test/rules.bzl",
        "def _rule_impl(ctx):",
        "    return []",
        "string_flag = rule(",
        "    implementation = _rule_impl,",
        "    build_setting = config.string()",
        ")",
        "def _transition_impl(settings, attr):",
        "    return {'//test:my_flag': 'custom string'}",
        "my_transition = transition(",
        "    implementation = _transition_impl,",
        "    inputs = [],",
        "    outputs = ['//test:my_flag'],",
        ")",
        "rule_with_deps_transition = rule(",
        "    implementation = _rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(cfg = my_transition),",
        "    }",
        ")",
        "simple_rule = rule(",
        "    implementation = _rule_impl,",
        "    attrs = {}",
        ")");
  }

  @Test
  public void testConfig_target() throws Exception {
    createConfigRulesAndBuild();

    assertThat(eval("config(//test:target_dep, target)")).isEqualTo(eval("//test:target_dep"));

    getHelper().setWholeTestUniverseScope("test:my_rule");

    assertThat(eval("config(//test:target_dep, target)")).isEqualTo(eval("//test:target_dep"));
    EvalThrowsResult execResult = evalThrows("config(//test:exec_dep, target)", true);
    assertThat(execResult.getMessage())
        .isEqualTo("No target (in) //test:exec_dep could be found in the 'target' configuration");
    assertConfigurableQueryCode(
        execResult.getFailureDetail(), ConfigurableQuery.Code.TARGET_MISSING);

    BuildConfigurationValue configuration =
        getConfiguration(Iterables.getOnlyElement(eval("config(//test:dep, target)")));

    assertThat(configuration).isNotNull();
    assertThat(configuration.isExecConfiguration()).isFalse();
    assertThat(configuration.isToolConfiguration()).isFalse();
  }

  @Test
  public void testConfig_nullConfig() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    assertThat(getConfiguration(Iterables.getOnlyElement(eval("config(//test:foo.java,null)"))))
        .isNull();
  }

  @Test
  public void testConfig_configHash() throws Exception {
    createConfigTransitioningRuleClass();
    writeFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'rule_with_deps_transition', 'simple_rule', 'string_flag')",
        "string_flag(",
        "    name = 'my_flag',",
        "    build_setting_default = '')",
        "rule_with_deps_transition(",
        "    name = 'buildme',",
        "    deps = [':mydep'])",
        "simple_rule(name = 'mydep')");

    // If we don't set --universe_scope=//test:buildme, cquery builds both //test:buildme and
    // //test:mydep as top-level targets. That means //test:mydep will have two configured targets:
    // one under the transitioned configuration and one under the top-level configuration. By
    // setting --universe_scope we ensure only the transitioned version exists.
    helper.setUniverseScope("//test:buildme");
    helper.setQuerySettings(Setting.ONLY_TARGET_DEPS, Setting.NO_IMPLICIT_DEPS);
    Set<CqueryNode> result = eval("deps(//test:buildme, 1)");
    assertThat(result).hasSize(2);

    ImmutableList<CqueryNode> stableOrderList = ImmutableList.copyOf(result);
    int myDepIndex = stableOrderList.get(0).getLabel().toString().equals("//test:mydep") ? 0 : 1;
    BuildConfigurationValue myDepConfig = getConfiguration(stableOrderList.get(myDepIndex));
    BuildConfigurationValue stringFlagConfig =
        getConfiguration(stableOrderList.get(1 - myDepIndex));

    // Note: eval() resets the universe scope after each call. We have to xplicitly set it again.
    helper.setUniverseScope("//test:buildme");
    assertThat(eval("config(//test:mydep, " + myDepConfig.checksum() + ")")).hasSize(1);

    helper.setUniverseScope("//test:buildme");
    QueryException e =
        assertThrows(
            QueryException.class,
            () -> eval("config(//test:mydep, " + stringFlagConfig.checksum() + ")"));
    assertThat(e)
        .hasMessageThat()
        .contains("No target (in) //test:mydep could be found in the configuration with checksum");
  }

  @Test
  public void testConfig_configHashPrefix() throws Exception {
    createConfigRulesAndBuild();
    writeFile("mytest/BUILD", "simple_rule(name = 'mytarget')");

    Set<CqueryNode> result = eval("//mytest:mytarget");
    String configHash = getConfiguration(Iterables.getOnlyElement(result)).checksum();
    String hashPrefix = configHash.substring(0, configHash.length() / 2);

    Set<CqueryNode> resultFromPrefix = eval("config(//mytest:mytarget," + hashPrefix + ")");
    assertThat(resultFromPrefix).containsExactlyElementsIn(result);
  }

  @Test
  public void testConfig_configHashUnknownPrefix() throws Exception {
    createConfigRulesAndBuild();
    writeFile("mytest/BUILD", "simple_rule(name = 'mytarget')");

    Set<CqueryNode> result = eval("//mytest:mytarget");
    String configHash = getConfiguration(Iterables.getOnlyElement(result)).checksum();
    String rightPrefix = configHash.substring(0, configHash.length() / 2);
    char lastChar = rightPrefix.charAt(rightPrefix.length() - 1);
    String wrongPrefix = rightPrefix.substring(0, rightPrefix.length() - 1) + (lastChar + 1);

    QueryException e =
        assertThrows(
            QueryException.class, () -> eval("config(//mytest:mytarget," + wrongPrefix + ")"));
    assertConfigurableQueryCode(
        e.getFailureDetail(), ConfigurableQuery.Code.INCORRECT_CONFIG_ARGUMENT_ERROR);
    assertThat(e)
        .hasMessageThat()
        .contains("config()'s second argument must identify a unique configuration");
  }

  @Test
  public void testConfig_exprArgumentFailure() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    EvalThrowsResult evalThrowsResult =
        evalThrows(
            "config(filter(\"??not-a-valid-regex\", //test:foo.java), null)",
            /* unconditionallyThrows= */ true);
    assertThat(evalThrowsResult.getMessage())
        .startsWith("illegal 'filter' pattern regexp '??not-a-valid-regex'");
    assertThat(evalThrowsResult.getFailureDetail().hasQuery()).isTrue();
    assertThat(evalThrowsResult.getFailureDetail().getQuery().getCode())
        .isEqualTo(Code.SYNTAX_ERROR);
  }

  @Test
  public void testExecTransitionNotFilteredByNoToolDeps() throws Exception {
    createConfigRulesAndBuild();
    helper.setQuerySettings(Setting.ONLY_TARGET_DEPS, Setting.NO_IMPLICIT_DEPS);
    assertThat(evalToListOfStrings("deps(//test:my_rule)"))
        .containsExactly("//test:my_rule", "//test:target_dep", "//test:dep");
  }

  @Test
  public void testRecursiveTargetPatternNeverThrowsError() throws Exception {
    Path parent =
        getHelper()
            .getScratch()
            .file("parent/BUILD", "sh_library(name = 'parent')")
            .getParentDirectory();
    Path child = parent.getRelative("child");
    child.createDirectory();
    Path badBuild = child.getRelative("BUILD");
    badBuild.createSymbolicLink(badBuild);

    helper.setKeepGoing(true);
    assertThat(eval("//parent:all")).isEqualTo(eval("//parent:parent"));

    helper.setKeepGoing(false);
    getHelper().turnOffFailFast();
    TargetParsingException e =
        assertThrows(TargetParsingException.class, () -> eval("//parent/..."));
    assertThat(e)
        .hasMessageThat()
        .isEqualTo(
            "error loading package under directory 'parent': no such package 'parent/child':"
                + " Symlink cycle detected while trying to find BUILD file"
                + " /workspace/parent/child/BUILD");
    assertThat(e.getDetailedExitCode().getFailureDetail().getPackageLoading().getCode())
        .isEqualTo(FailureDetails.PackageLoading.Code.SYMLINK_CYCLE_OR_INFINITE_EXPANSION);
  }

  // Regression test for b/175739699
  @Test
  public void testRecursiveTargetPatternOutsideOfScopeFailsGracefully() throws Exception {
    writeFile("testA/BUILD", "sh_library(name = 'testA')");
    writeFile("testB/BUILD", "sh_library(name = 'testB')");
    writeFile("testB/testC/BUILD", "sh_library(name = 'testC')");
    helper.setUniverseScope("//testA");
    QueryException e = assertThrows(QueryException.class, () -> eval("//testB/..."));
    assertThat(e.getFailureDetail().getQuery().getCode())
        .isEqualTo(Query.Code.TARGET_NOT_IN_UNIVERSE_SCOPE);
    assertThat(e).hasMessageThat().contains("package is not in scope");
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<CqueryNode> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<CqueryNode> resultIterator = result.iterator();
    CqueryNode first = resultIterator.next();
    if (first.getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }

  @Test
  public void testSomePath_depInCustomConfiguration() throws Exception {
    createConfigTransitioningRuleClass();
    writeFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'rule_with_deps_transition', 'simple_rule', 'string_flag')",
        "string_flag(",
        "    name = 'my_flag',",
        "    build_setting_default = '')",
        "rule_with_deps_transition(",
        "    name = 'buildme',",
        "    deps = [':mydep'])",
        "simple_rule(name = 'mydep')");

    // If we don't set --universe_scope=//test:buildme, then cquery builds both //test:buildme and
    // //test:mydep as top-level targets. That means //test:mydep will have two configured targets:
    // one under the transitioned configuration and one under the top-level configuration. In these
    // cases cquery prefers the top-level configured one, which won't produce a match since that's
    // not the one down this dependency path.
    helper.setUniverseScope("//test:buildme");
    Set<CqueryNode> result = eval("somepath(//test:buildme, //test:mydep)");
    assertThat(result.stream().map(kct -> kct.getLabel().toString()).collect(Collectors.toList()))
        .contains("//test:mydep");
  }

  /** Return an empty BuildOptions for testing fragment dropping. * */
  public static class RemoveTestOptionsTransition implements PatchTransition {
    @Override
    public ImmutableSet<Class<? extends FragmentOptions>> requiresOptionFragments() {
      return ImmutableSet.of(TestOptions.class);
    }

    @Override
    public BuildOptions patch(BuildOptionsView options, EventHandler eventHandler) {
      BuildOptions.Builder builder = BuildOptions.builder();
      for (FragmentOptions option : options.underlying().getNativeOptions()) {
        if (!(option instanceof TestOptions)) {
          builder.addFragmentOptions(option);
        }
      }
      // This does not copy over Starlark options!!
      return builder.build();
    }
  }

  @Test
  public void testQueryHandlesDroppingFragments() throws Exception {
    MockRule ruleDropOptions =
        () ->
            MockRule.define(
                "rule_drop_options",
                attr("dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(TransitionFactories.of(new RemoveTestOptionsTransition())));
    MockRule simpleRule =
        () ->
            MockRule.define(
                "simple_rule", attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE));

    helper.useRuleClassProvider(setRuleClassProviders(ruleDropOptions, simpleRule).build());
    writeFile(
        "test/BUILD",
        "rule_drop_options(name = 'top', dep = ':foo')",
        "simple_rule(name='foo', deps = [':bar'])",
        "simple_rule(name='bar')");

    Set<CqueryNode> result = eval("somepath(//test:top, filter(//test:bar, deps(//test:top)))");
    assertThat(result).isNotEmpty();
  }

  @Test
  public void testLabelExpressionsMatchesAllConfiguredTargetsWithLabel() throws Exception {
    createConfigTransitioningRuleClass();
    writeFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'rule_with_deps_transition', 'simple_rule', 'string_flag')",
        "string_flag(",
        "    name = 'my_flag',",
        "    build_setting_default = '')",
        "rule_with_deps_transition(",
        "    name = 'transitioner',",
        "    deps = [':simple'])",
        "simple_rule(name = 'simple')");

    helper.setUniverseScope("//test:transitioner,//test:simple");
    Set<CqueryNode> result = eval("//test:simple");
    assertThat(result.size()).isEqualTo(2);
  }

  @Test
  public void testConfigFunctionRefinesMultipleMatches() throws Exception {
    // Peer to testLabelExpressionsMatchesAllConfiguredTargetsWithLabel. The point of that test is
    // to show "cquery //foo:bar" might return multiple configured targets. The point of this test
    // is to show that config() can refine the same query to a specific one.
    createConfigTransitioningRuleClass();
    writeFile(
        "test/BUILD",
        "load('//test:rules.bzl', 'rule_with_deps_transition', 'simple_rule', 'string_flag')",
        "string_flag(",
        "    name = 'my_flag',",
        "    build_setting_default = '')",
        "rule_with_deps_transition(",
        "    name = 'transitioner',",
        "    deps = [':simple'])",
        "simple_rule(name = 'simple')");

    helper.setUniverseScope("//test:transitioner,//test:simple");
    Set<CqueryNode> result = eval("config(//test:simple, target)");
    assertThat(result.size()).isEqualTo(1);
  }

  @Test
  public void testAspectDepsAppearInCqueryDeps() throws Exception {
    writeFile(
        "donut/test.bzl",
        "TestAspectInfo = provider('TestAspectInfo', fields = ['info'])",
        "def _test_aspect_impl(target, ctx):",
        "    return [",
        "        TestAspectInfo(",
        "            info = depset([target.label]),",
        "        ),",
        "    ]",
        "",
        "_test_aspect = aspect(",
        "    implementation = _test_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    attrs = {",
        "        '_test_attr': attr.label(",
        "            allow_files = True,",
        "            default = Label('//donut:test_filegroup'),",
        "        ),",
        "    },",
        "    provides = [TestAspectInfo],",
        ")",
        "def _test_impl(ctx):",
        "    pass",
        "test_rule = rule(",
        "    _test_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(",
        "            aspects = [_test_aspect],",
        "        ),",
        "    },",
        ")");
    writeFile(
        "donut/BUILD",
        "load(':test.bzl', 'test_rule')",
        "filegroup(",
        "    name = 'test_filegroup',",
        "    srcs = ['test.bzl'],",
        ")",
        "test_rule(",
        "    name = 'test_rule_dep',",
        ")",
        "test_rule(",
        "    name = 'test_rule',",
        "    deps = [':test_rule_dep'],",
        ")");

    helper.setQuerySettings(Setting.INCLUDE_ASPECTS, Setting.EXPLICIT_ASPECTS);
    var result =
        eval("filter(//donut, deps(//donut:test_rule))").stream()
            .map(cf -> cf.getDescription(LabelPrinter.legacy()))
            .collect(ImmutableList.toImmutableList());
    assertThat(result)
        .containsExactly(
            "//donut:test_rule",
            "//donut:test_rule_dep",
            "//donut:test.bzl%_test_aspect of //donut:test_rule_dep",
            "//donut:test.bzl",
            "//donut:test_filegroup");
  }

  @Test
  public void testAspectOnAspectDepsAppearInCqueryDeps() throws Exception {
    writeFile(
        "donut/test.bzl",
        "TestAspectInfo = provider('TestAspectInfo', fields = ['info'])",
        "TestAspectOnAspectInfo = provider('TestAspectOnAspectInfo', fields = ['info'])",
        "def _test_aspect_impl(target, ctx):",
        "    return [",
        "        TestAspectInfo(",
        "            info = depset([target.label]),",
        "        ),",
        "    ]",
        "_test_aspect = aspect(",
        "    implementation = _test_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    attrs = {",
        "        '_test_attr': attr.label(",
        "            allow_files = True,",
        "            default = Label('//donut:test_aspect_filegroup'),",
        "        ),",
        "    },",
        "    provides = [TestAspectInfo],",
        ")",
        "def _test_aspect_on_aspect_impl(target, ctx):",
        "    return [",
        "        TestAspectOnAspectInfo(",
        "            info = depset(",
        "                direct = [target.label],",
        "                transitive = [target[TestAspectInfo].info],",
        "            ),",
        "        ),",
        "    ]",
        "_test_aspect_on_aspect = aspect(",
        "    implementation = _test_aspect_on_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    attrs = {",
        "        '_test_attr': attr.label(",
        "            allow_files = True,",
        "            default = Label('//donut:test_aspect_on_aspect_filegroup'),",
        "        ),",
        "    },",
        "    required_aspect_providers = [TestAspectInfo],",
        "    provides = [TestAspectOnAspectInfo],",
        ")",
        "def _test_impl(ctx):",
        "    pass",
        "test_rule = rule(",
        "    _test_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(",
        "            aspects = [_test_aspect],",
        "        ),",
        "    },",
        ")",
        "def _test_aspect_on_aspect_rule_impl(ctx):",
        "    pass",
        "test_aspect_on_aspect_rule = rule(",
        "    _test_aspect_on_aspect_rule_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(",
        "            aspects = [_test_aspect, _test_aspect_on_aspect],",
        "        ),",
        "    },",
        ")");
    writeFile("donut/test_aspect.file");
    writeFile("donut/test_aspect_on_aspect.file");
    writeFile(
        "donut/BUILD",
        "load(':test.bzl', 'test_rule', 'test_aspect_on_aspect_rule')",
        "filegroup(",
        "    name = 'test_aspect_filegroup',",
        "    srcs = ['test_aspect.file'],",
        ")",
        "filegroup(",
        "    name = 'test_aspect_on_aspect_filegroup',",
        "    srcs = ['test_aspect_on_aspect.file'],",
        ")",
        "test_rule(",
        "    name = 'test_rule_dep',",
        ")",
        "test_rule(",
        "    name = 'test_rule',",
        "    deps = [':test_rule_dep'],",
        ")",
        "test_aspect_on_aspect_rule(",
        "    name = 'test_aspect_on_aspect_rule',",
        "    deps = ['test_rule'],",
        ")");

    helper.setUniverseScope("//donut/...");
    helper.setQuerySettings(Setting.INCLUDE_ASPECTS, Setting.EXPLICIT_ASPECTS);
    var result =
        eval("filter(//donut, deps(//donut:test_aspect_on_aspect_rule))").stream()
            .map(cf -> cf.getDescription(LabelPrinter.legacy()))
            .collect(toImmutableList());
    assertThat(result)
        .containsExactly(
            "//donut:test.bzl%_test_aspect_on_aspect on top of"
                + " [//donut:test.bzl%_test_aspect of //donut:test_rule_dep]",
            "//donut:test.bzl%_test_aspect_on_aspect on top of"
                + " [//donut:test.bzl%_test_aspect of //donut:test_rule]",
            "//donut:test_rule_dep",
            "//donut:test_rule",
            "//donut:test.bzl%_test_aspect of //donut:test_rule_dep",
            "//donut:test.bzl%_test_aspect of //donut:test_rule",
            "//donut:test_aspect_on_aspect_rule",
            "//donut:test_aspect.file",
            "//donut:test_aspect_on_aspect_filegroup",
            "//donut:test_aspect_on_aspect.file",
            "//donut:test_aspect_filegroup");
  }

  @Test
  public void testAspectDepsAppearInCqueryRdeps() throws Exception {
    writeFile(
        "donut/test.bzl",
        "TestAspectInfo = provider('TestAspectInfo', fields = ['info'])",
        "def _test_aspect_impl(target, ctx):",
        "    return [",
        "        TestAspectInfo(",
        "            info = depset([target.label]),",
        "        ),",
        "    ]",
        "",
        "_test_aspect = aspect(",
        "    implementation = _test_aspect_impl,",
        "    attr_aspects = ['deps'],",
        "    attrs = {",
        "        '_test_attr': attr.label(",
        "            allow_files = True,",
        "            default = Label('//donut:test_filegroup'),",
        "        ),",
        "    },",
        "    provides = [TestAspectInfo],",
        ")",
        "def _test_impl(ctx):",
        "    pass",
        "test_rule = rule(",
        "    _test_impl,",
        "    attrs = {",
        "        'deps': attr.label_list(",
        "            aspects = [_test_aspect],",
        "        ),",
        "    },",
        ")");
    writeFile(
        "donut/BUILD",
        "load(':test.bzl', 'test_rule')",
        "filegroup(",
        "    name = 'test_filegroup',",
        "    srcs = ['test.bzl'],",
        ")",
        "test_rule(",
        "    name = 'test_rule_dep',",
        ")",
        "test_rule(",
        "    name = 'test_rule',",
        "    deps = [':test_rule_dep'],",
        ")");

    helper.setQuerySettings(Setting.INCLUDE_ASPECTS, Setting.EXPLICIT_ASPECTS);
    var result =
        eval("rdeps(//donut/..., //donut:test_filegroup)").stream()
            .map(cf -> cf.getDescription(LabelPrinter.legacy()))
            .collect(toImmutableList());
    assertThat(result)
        .containsExactly(
            "//donut:test_filegroup",
            "//donut:test_rule",
            "//donut:test.bzl%_test_aspect of //donut:test_rule_dep");
  }
}
