// Copyright 2019 The Bazel Authors. All rights reserved.
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

import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.config.transitions.NoTransition;
import com.google.devtools.build.lib.analysis.config.transitions.TransitionFactory;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.cmdline.RepositoryMapping;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.RuleTransitionData;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions.Transitions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Tests for the transitions output format. */
@RunWith(JUnit4.class)
public class TransitionsOutputFormatterTest extends ConfiguredTargetQueryTest {

  private CqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();
  private ConfiguredRuleClassProvider ruleClassProvider;

  @Before
  public final void setUpCqueryOptions() {
    this.options = new CqueryOptions();
    helper.setQuerySettings(Setting.INCLUDE_ASPECTS);
    this.reporter = new Reporter(new EventBus(), events::add);
  }

  @Test
  public void testTransitions_full() throws Exception {
    setUpRules();

    writeFile(
        "test/BUILD",
        "my_rule(name = 'rule_with_patch',",
        "  patched = [':foo', ':foo2', ':trimmed_foo'],",
        ")",
        "my_rule(name = 'rule_with_split',",
        "  split = ':bar',",
        ")",
        "simple_rule(name = 'foo')",
        "simple_rule(name = 'foo2')",
        "simple_rule(name = 'trimmed_foo')",
        "simple_rule(name = 'bar')");

    List<String> result = getOutput("deps(//test:rule_with_patch)", Transitions.FULL);

    assertThat(result.get(0)).startsWith("FooPatchTransition -> //test:rule_with_patch");
    assertThat(result.get(1)).startsWith("  patched#//test:foo#FooPatch");
    assertThat(result.get(2)).isEqualTo("    foo:SET BY RULE CLASS PATCH -> [SET BY PATCH]");
    assertThat(result.get(3)).startsWith("  patched#//test:foo2#FooPatch");
    assertThat(result.get(4)).isEqualTo(result.get(2));
    assertThat(result.get(5))
        .startsWith("  patched#//test:trimmed_foo#(FooPatchTransition + FooPatchTransition(trim))");
    assertThat(result.get(6)).isEqualTo("    foo:SET BY RULE CLASS PATCH -> [SET BY TRIM]");

    result = getOutput("deps(//test:rule_with_split)", Transitions.FULL);
    assertThat(result.get(1)).startsWith("  split#//test:bar#FooSplitTransition");
    // TODO(shahan): the right hand side of the diff below is in split dep ordering, which is
    // dependent on checksum values. It could be brittle.
    assertThat(result.get(2))
        .isEqualTo("    foo:SET BY RULE CLASS PATCH -> [SET BY SPLIT 2, SET BY SPLIT 1]");
  }

  @Test
  public void testTransitions_lite() throws Exception {
    setUpRules();

    writeFile(
        "test/BUILD",
        "my_rule(name = 'rule_with_patch',",
        "  patched = [':foo', ':foo2'],",
        ")",
        "my_rule(name = 'rule_with_split',",
        "  split = ':bar',",
        ")",
        "simple_rule(name = 'foo')",
        "simple_rule(name = 'foo2')",
        "simple_rule(name = 'bar')");

    List<String> result = getOutput("deps(//test:rule_with_patch)", Transitions.LITE);

    assertThat(result.get(0)).startsWith("FooPatchTransition -> //test:rule_with_patch");
    assertThat(result.get(1)).startsWith("  patched#//test:foo#FooPatchTransition");
    assertThat(result.get(2)).startsWith("  patched#//test:foo2#FooPatchTransition");

    result = getOutput("deps(//test:rule_with_split)", Transitions.LITE);
    assertThat(result.get(0)).startsWith("FooPatchTransition -> //test:rule_with_split");
    assertThat(result.get(1)).startsWith("  split#//test:bar#FooSplitTransition");
  }

  @Test
  public void testTransitions_getRightConfigurations() throws Exception {
    setUpRules();

    writeFile(
        "test/BUILD",
        "my_rule(name = 'rule_with_patch',",
        "  patched = [':foo'],",
        ")",
        "simple_rule(name = 'foo')");

    List<String> result = getOutput("deps(//test:rule_with_patch)", Transitions.LITE);
    String depEntry = result.get(2);
    // depEntry is "//test:rule_with_path (<config_id>)". This gets just "<config_id>".
    String postPatchConfig =
        depEntry.substring(depEntry.lastIndexOf("(") + 1, depEntry.length() - 1);
    assertThat(result.get(1)).endsWith(postPatchConfig);
  }

  @Test
  public void testTransitions_noTransitions() throws Exception {
    setUpRules();
    writeFile(
        "test/BUILD",
        "simple_rule(name = 'foo')");

    List<String> result = getOutput("//test:foo", Transitions.NONE);
    assertThat(result).isEmpty();
    assertThat(events).hasSize(1);
    assertThat(events.get(0).getMessage())
        .isEqualTo(
            "Instead of using --output=transitions, set the --transitions flag explicitly to 'lite'"
                + " or 'full'");
  }

  @Test
  public void nonAttributeDependencySkipped() throws Exception {
    setUpRules();

    // A visibility dependency on a package_group produces a
    // DependencyKind.NonAttributeDependencyKind. This test checks that the existence of those
    // attribute types doesn't crash cquery.

    writeFile(
        "test/BUILD",
        "package_group(",
        "    name = 'custom_visibility',",
        "    packages = ['//test/...'],",
        ")",
        "simple_rule(",
        "    name = 'child',",
        ")",
        "simple_rule(",
        "    name = 'parent',",
        "    deps = [':child'],",
        "    visibility = [':custom_visibility'],",
        ")");

    assertThat(getOutput("deps(//test:parent)", Transitions.LITE)).isNotNull();
    assertThat(events).isEmpty();
  }

  private void setUpRules() throws Exception {
    TransitionFactory<RuleTransitionData> infixTrimmingTransitionFactory =
        (ruleData) -> {
          if (!ruleData.rule().getName().contains("trimmed")) {
            return NoTransition.INSTANCE;
          }
          // rename the transition so it's distinguishable from the others in tests
          return new FooPatchTransition("SET BY TRIM", "FooPatchTransition(trim)");
        };
    FooPatchTransition ruleClassTransition = new FooPatchTransition("SET BY RULE CLASS PATCH");
    FooPatchTransition attributePatchTransition = new FooPatchTransition("SET BY PATCH");
    FooSplitTransition attributeSplitTransitions =
        new FooSplitTransition("SET BY SPLIT 1", "SET BY SPLIT 2");

    MockRule ruleWithTransitions =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder
                        .cfg(ruleClassTransition)
                        .add(
                            attr("patched", LABEL_LIST)
                                .allowedFileTypes(FileTypeSet.ANY_FILE)
                                .cfg(TransitionFactories.of(attributePatchTransition)))
                        .add(
                            attr("split", LABEL)
                                .allowedFileTypes(FileTypeSet.ANY_FILE)
                                .cfg(TransitionFactories.of(attributeSplitTransitions))));
    MockRule simpleRule =
        () ->
            MockRule.define(
                "simple_rule",
                (builder, env) ->
                    builder
                        .add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));

    this.ruleClassProvider =
        setRuleClassProviders(ruleWithTransitions, simpleRule)
            .overrideTrimmingTransitionFactoryForTesting(infixTrimmingTransitionFactory)
            .build();
    helper.useRuleClassProvider(ruleClassProvider);
  }

  private List<String> getOutput(String queryExpression, CqueryOptions.Transitions verbosity)
      throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<ConfiguredTarget> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);
    options.transitions = verbosity;
    // TODO(blaze-configurability): Test late-bound attributes.
    TransitionsOutputFormatterCallback callback =
        new TransitionsOutputFormatterCallback(
            reporter,
            options,
            /* out= */ null,
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            ruleClassProvider,
            RepositoryMapping.ALWAYS_FALLBACK);
    env.evaluateQuery(env.transformParsedQuery(QueryParser.parse(queryExpression, env)), callback);
    return callback.getResult();
  }
}
