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

import static com.google.common.truth.Truth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static com.google.devtools.build.lib.packages.Type.STRING;
import static com.google.devtools.build.lib.testutil.MoreAsserts.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.config.FragmentOptions;
import com.google.devtools.build.lib.analysis.config.HostTransition;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.PatchTransition;
import com.google.devtools.build.lib.analysis.config.transitions.SplitTransition;
import com.google.devtools.build.lib.analysis.test.TestConfiguration.TestOptions;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.packages.BuildType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryException;
import com.google.devtools.build.lib.query2.testutil.PostAnalysisQueryTest;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.devtools.build.lib.vfs.Path;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/**
 * Tests for {@link ConfiguredTargetQueryEnvironment}.
 *
 * <p>TODO(juliexxia): separate out tests in this file into one test per tested functionality.
 */
@RunWith(JUnit4.class)
public class ConfiguredTargetQueryTest extends PostAnalysisQueryTest<ConfiguredTarget> {

  @Override
  protected QueryHelper<ConfiguredTarget> createQueryHelper() {
    if (helper != null) {
      getHelper().cleanUp();
    }
    helper = new ConfiguredTargetQueryHelper();
    return helper;
  }

  @Override
  public HashMap<String, QueryFunction> getDefaultFunctions() {
    ImmutableList<QueryFunction> defaultFunctions =
        ImmutableList.copyOf(ConfiguredTargetQueryEnvironment.FUNCTIONS);
    HashMap<String, QueryFunction> functions = new HashMap<>();
    for (QueryFunction queryFunction : defaultFunctions) {
      functions.put(queryFunction.getName(), queryFunction);
    }
    return functions;
  }

  @Override
  protected final BuildConfiguration getConfiguration(ConfiguredTarget ct) {
    return getHelper()
        .getSkyframeExecutor()
        .getConfiguration(getHelper().getReporter(), ct.getConfigurationKey());
  }

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
                    .cfg(TransitionFactories.of(new TestArgPatchTransition("SET BY PATCH"))),
                attr("string_dep", STRING),
                attr("split_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(
                        TransitionFactories.of(
                            new TestArgSplitTransition("SET BY SPLIT 1", "SET BY SPLIT 2"))),
                attr("patch_dep_list", LABEL_LIST)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(TransitionFactories.of(new TestArgPatchTransition("SET BY PATCH 2"))));
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
    ConfiguredTarget patchDep =
        Iterables.getOnlyElement(eval("labels('patch_dep', //test:my_rule)"));
    ConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();
    assertThat(patchDep.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
  }

  @Test
  public void testLabelsFunction_splitTransitionAttribute() throws Exception {
    setUpLabelsFunctionTests();

    ConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();

    Collection<ConfiguredTarget> splitDeps = eval("labels('split_dep', //test:my_rule)");
    assertThat(splitDeps).hasSize(2);
    for (ConfiguredTarget ct : splitDeps) {
      assertThat(ct.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
    }
  }

  @Test
  public void testLabelsFunction_labelListAttribute() throws Exception {
    setUpLabelsFunctionTests();

    ConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();

    // Test that this works for label_lists as well.
    Set<ConfiguredTarget> deps = eval("labels('patch_dep_list', //test:my_rule)");
    assertThat(deps).hasSize(2);
    for (ConfiguredTarget ct : deps) {
      assertThat(ct.getConfigurationChecksum()).doesNotMatch(targetConfiguration);
    }
  }

  @Test
  public void testLabelsFunction_errorsOnBadAttribute() throws Exception {
    setUpLabelsFunctionTests();

    ConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    String targetConfiguration = myRule.getConfigurationChecksum();

    // Test that the proper error is thrown when requesting an attribute that doesn't exist.
    assertThat(evalThrows("labels('fake_attr', //test:my_rule)", true))
        .isEqualTo(
            String.format(
                "in 'fake_attr' of rule //test:my_rule:  ConfiguredTarget(//test:my_rule, %s) "
                    + "of type rule_with_transitions does not have attribute 'fake_attr'",
                targetConfiguration));
  }

  @Test
  public void testLabelsFunction_nonLabelAttribute() throws Exception {
    setUpLabelsFunctionTests();
    assertThat(eval("labels('string_dep', //test:my_rule)")).isEmpty();
  }

  @Test
  public void testAlias_filtering() throws Exception {
    MockRule ruleWithHostDep =
        () ->
            MockRule.define(
                "rule_with_host_dep",
                attr("host_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(HostTransition.createFactory()),
                attr("$impl_dep", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .value(Label.parseAbsoluteUnchecked("//test:other")));
    MockRule simpleRule = () -> MockRule.define("simple_rule");

    helper.useRuleClassProvider(setRuleClassProviders(ruleWithHostDep, simpleRule).build());
    writeFile(
        "test/BUILD",
        "alias(name = 'other_my_rule', actual = ':my_rule')",
        "rule_with_host_dep(name = 'my_rule', host_dep = ':host_dep')",
        "alias(name = 'other_host_dep', actual = ':host_dep')",
        "simple_rule(name='host_dep')",
        "alias(name = 'other_impl_dep', actual = 'impl_dep')",
        "simple_rule(name='impl_dep')");

    ConfiguredTarget other = Iterables.getOnlyElement(eval("//test:other_my_rule"));
    ConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));
    // Note: {@link AliasConfiguredTarget#getLabel} returns the label of the "actual" value not the
    // label of the alias.
    assertThat(other.getLabel()).isEqualTo(myRule.getLabel());

    // Regression test for b/73496081 in which alias-ed configured targets were skipping filtering.
    helper.setQuerySettings(Setting.ONLY_TARGET_DEPS, Setting.NO_IMPLICIT_DEPS);
    assertThat(evalToListOfStrings("deps(//test:other_my_rule)-//test:other_my_rule"))
        .isEqualTo(evalToListOfStrings("//test:my_rule"));
  }

  @Test
  public void testTopLevelTransition() throws Exception {
    MockRule ruleClassTransition =
        () ->
            MockRule.define(
                "rule_class_transition",
                (builder, env) -> builder.cfg(new TestArgPatchTransition("SET BY PATCH")).build());

    helper.useRuleClassProvider(setRuleClassProviders(ruleClassTransition).build());
    helper.setUniverseScope("//test:rule_class");

    writeFile("test/BUILD", "rule_class_transition(name='rule_class')");

    Set<ConfiguredTarget> ruleClass = eval("//test:rule_class");
    TestOptions testOptions =
        getConfiguration(Iterables.getOnlyElement(ruleClass)).getOptions().get(TestOptions.class);
    assertThat(testOptions.testArguments).containsExactly("SET BY PATCH");
  }

  /** SplitTransition on --test_arg */
  public static class TestArgSplitTransition implements SplitTransition {
    String toOption1;
    String toOption2;

    public TestArgSplitTransition(String toOption1, String toOptions2) {
      this.toOption1 = toOption1;
      this.toOption2 = toOptions2;
    }

    @Override
    public List<BuildOptions> split(BuildOptions options) {
      BuildOptions result1 = options.clone();
      BuildOptions result2 = options.clone();
      result1.get(TestOptions.class).testArguments = Collections.singletonList(toOption1);
      result2.get(TestOptions.class).testArguments = Collections.singletonList(toOption2);
      return ImmutableList.of(result1, result2);
    }
  }

  private void createConfigRulesAndBuild() throws Exception {
    MockRule ruleWithTransitions =
        () ->
            MockRule.define(
                "my_rule",
                attr("target", LABEL).allowedFileTypes(FileTypeSet.ANY_FILE),
                attr("host", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(HostTransition.createFactory()),
                attr("exec", LABEL)
                    .allowedFileTypes(FileTypeSet.ANY_FILE)
                    .cfg(ExecutionTransitionFactory.create()),
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
        "  host = ':host_dep',",
        "  exec = ':exec_dep',",
        "  deps = [':dep'],",
        ")",
        "simple_rule(name = 'target_dep', dep=':dep')",
        "simple_rule(name = 'host_dep', dep=':dep')",
        "simple_rule(name = 'exec_dep', dep=':dep')",
        "simple_rule(name = 'dep')");
  }

  private void createConfigTransitioningRuleClass() throws Exception {
    writeFile(
        "tools/whitelists/function_transition_whitelist/BUILD",
        "package_group(",
        "    name = 'function_transition_whitelist',",
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
        "        '_whitelist_function_transition': attr.label(",
        "            default = '//tools/whitelists/function_transition_whitelist',",
        "        ),",
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
    assertThat(evalThrows("config(//test:host_dep, target)", true))
        .isEqualTo("No target (in) //test:host_dep could be found in the 'target' configuration");
    assertThat(evalThrows("config(//test:exec_dep, target)", true))
        .isEqualTo("No target (in) //test:exec_dep could be found in the 'target' configuration");

    BuildConfiguration configuration =
        getConfiguration(Iterables.getOnlyElement(eval("config(//test:dep, target)")));

    assertThat(configuration).isNotNull();
    assertThat(configuration.isHostConfiguration()).isFalse();
    assertThat(configuration.isExecConfiguration()).isFalse();
    assertThat(configuration.isToolConfiguration()).isFalse();
  }

  @Test
  public void testConfig_hostTransition() throws Exception {
    createConfigRulesAndBuild();

    getHelper().setWholeTestUniverseScope("test:my_rule");

    assertThat(evalThrows("config(//test:target_dep, host)", true))
        .isEqualTo("No target (in) //test:target_dep could be found in the 'host' configuration");
    assertThat(eval("config(//test:host_dep, host)")).isEqualTo(eval("//test:host_dep"));
    assertThat(evalThrows("config(//test:exec_dep, host)", true))
        .isEqualTo("No target (in) //test:exec_dep could be found in the 'host' configuration");

    BuildConfiguration configuration =
        getConfiguration(Iterables.getOnlyElement(eval("config(//test:dep, host)")));

    assertThat(configuration).isNotNull();
    assertThat(configuration.isHostConfiguration()).isTrue();
    assertThat(configuration.isExecConfiguration()).isFalse();
    assertThat(configuration.isToolConfiguration()).isTrue();
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
    Set<ConfiguredTarget> result = eval("deps(//test:buildme, 1)");
    assertThat(result).hasSize(2);

    ImmutableList<ConfiguredTarget> stableOrderList = ImmutableList.copyOf(result);
    int myDepIndex = stableOrderList.get(0).getLabel().toString().equals("//test:mydep") ? 0 : 1;
    BuildConfiguration myDepConfig = getConfiguration(stableOrderList.get(myDepIndex));
    BuildConfiguration stringFlagConfig = getConfiguration(stableOrderList.get(1 - myDepIndex));

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
  public void testConfig_badConfig() throws Exception {
    createConfigRulesAndBuild();
    assertThat(evalThrows("config(//test:my_rule,foo)", true))
        .isEqualTo(
            "Unknown value 'foo'. The second argument of config() must be 'target', 'host', "
                + "'null', or a valid configuration hash (i.e. one of the outputs of "
                + "'blaze config')");
  }

  @Test
  public void testExecTransitionNotFilteredByNoHostDeps() throws Exception {
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
    assertThat(evalThrows("//parent/...", true))
        .isEqualTo(
            "no such package 'parent/child': Symlink cycle detected while trying to "
                + "find BUILD file /workspace/parent/child/BUILD");
  }

  @Override
  @Test
  public void testMultipleTopLevelConfigurations_nullConfigs() throws Exception {
    writeFile("test/BUILD", "java_library(name='my_java',", "  srcs = ['foo.java'],", ")");

    Set<ConfiguredTarget> result = eval("//test:my_java+//test:foo.java");

    assertThat(result).hasSize(2);

    Iterator<ConfiguredTarget> resultIterator = result.iterator();
    ConfiguredTarget first = resultIterator.next();
    if (first.getLabel().toString().equals("//test:foo.java")) {
      assertThat(getConfiguration(first)).isNull();
      assertThat(getConfiguration(resultIterator.next())).isNotNull();
    } else {
      assertThat(getConfiguration(first)).isNotNull();
      assertThat(getConfiguration(resultIterator.next())).isNull();
    }
  }

  @Test
  public void testSomePath_DepInCustomConfiguration() throws Exception {
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
    Set<ConfiguredTarget> result = eval("somepath(//test:buildme, //test:mydep)");
    assertThat(result.stream().map(ct -> ct.getLabel().toString()).collect(Collectors.toList()))
        .contains("//test:mydep");
  }

  /** Return an empty BuildOptions for testing fragment dropping. * */
  public static class RemoveTestOptionsTransition implements PatchTransition {
    @Override
    public BuildOptions patch(BuildOptions options) {
      BuildOptions.Builder builder = BuildOptions.builder();
      for (FragmentOptions option : options.getNativeOptions()) {
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

    Set<ConfiguredTarget> result =
        eval("somepath(//test:top, filter(//test:bar, deps(//test:top)))");
    assertThat(result).isNotEmpty();
  }
}
