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
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;

import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.config.TransitionFactories;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.ProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver.Mode;
import com.google.devtools.build.lib.util.FileTypeSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import org.junit.Before;
import org.junit.Test;

/**
 * Test for cquery's proto output format.
 *
 * <p>TODO(blaze-configurability): refactor all cquery output format tests to consolidate duplicate
 * infrastructure.
 */
public class ProtoOutputFormatterCallbackTest extends ConfiguredTargetQueryTest {

  private CqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  @Before
  public final void setUpCqueryOptions() {
    this.options = new CqueryOptions();
    // TODO(bazel-team): reduce the confusion about these two seemingly similar settings.
    // options.aspectDeps impacts how proto and similar output formatters output aspect results.
    // Setting.INCLUDE_ASPECTS impacts whether or not aspect dependencies are included when
    // following target deps. See CommonQueryOptions for further flag details.
    options.aspectDeps = Mode.OFF;
    helper.setQuerySettings(Setting.INCLUDE_ASPECTS);
    options.protoIncludeConfigurations = true;
    options.protoIncludeRuleInputsAndOutputs = true;
    this.reporter = new Reporter(new EventBus(), events::add);
  }

  @Test
  public void testSelectInAttribute() throws Exception {
    MockRule depsRule =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder
                        .add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));
    helper.useRuleClassProvider(setRuleClassProviders(depsRule).build());

    writeFile(
        "test/BUILD",
        "my_rule(name = 'my_rule',",
        "  deps = select({",
        "    ':garfield': ['lasagna.java', 'naps.java'],",
        "    '//conditions:default': ['mondays.java']",
        "  })",
        ")",
        "config_setting(",
        "  name = 'garfield',",
        "  values = {'test_arg': 'cat'}",
        ")");

    AnalysisProtos.ConfiguredTarget myRuleProto =
        (Iterables.getOnlyElement(getOutput("//test:my_rule").getResultsList()));
    List<Build.Attribute> attributes = myRuleProto.getTarget().getRule().getAttributeList();
    for (Build.Attribute attribute : attributes) {
      if (!attribute.getName().equals("deps")) {
        continue;
      }
      assertThat(attribute.getStringListValueCount()).isEqualTo(1);
      assertThat(attribute.getStringListValue(0)).isEqualTo("//test:mondays.java");
      break;
    }

    getHelper().useConfiguration("--test_arg=cat");
    myRuleProto = Iterables.getOnlyElement(getOutput("//test:my_rule").getResultsList());
    attributes = myRuleProto.getTarget().getRule().getAttributeList();
    for (Build.Attribute attribute : attributes) {
      if (!attribute.getName().equals("deps")) {
        continue;
      }
      assertThat(attribute.getStringListValueCount()).isEqualTo(2);
      assertThat(attribute.getStringListValue(0)).isEqualTo("//test:lasagna.java");
      assertThat(attribute.getStringListValue(1)).isEqualTo("//test:naps.java");
      break;
    }
  }

  @Test
  public void testConfigurationHash() throws Exception {
    TestArgPatchTransition attributePatchTransition = new TestArgPatchTransition("SET BY PATCH");

    MockRule ruleWithPatch =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder.add(
                        attr("deps", LABEL_LIST)
                            .allowedFileTypes(FileTypeSet.ANY_FILE)
                            .cfg(TransitionFactories.of(attributePatchTransition))));

    helper.useRuleClassProvider(setRuleClassProviders(ruleWithPatch, getSimpleRule()).build());

    writeFile(
        "test/BUILD",
        "my_rule(name = 'my_rule',",
        "  deps = [':patched'],",
        ")",
        "simple_rule(name = 'dep')");

    // Assert checksum from proto is proper checksum.
    AnalysisProtos.ConfiguredTarget myRuleProto =
        Iterables.getOnlyElement(getOutput("//test:my_rule").getResultsList());
    KeyedConfiguredTarget myRule = Iterables.getOnlyElement(eval("//test:my_rule"));

    assertThat(myRuleProto.getConfiguration().getChecksum())
        .isEqualTo(myRule.getConfigurationChecksum());

    // Assert checksum for two configured targets in proto are not the same.
    List<AnalysisProtos.ConfiguredTarget> protoDeps =
        getOutput("deps(//test:my_rule)").getResultsList();
    assertThat(protoDeps).hasSize(2);

    Iterator<AnalysisProtos.ConfiguredTarget> protoDepsIterator = protoDeps.iterator();
    assertThat(protoDepsIterator.next().getConfiguration().getChecksum())
        .isNotEqualTo(protoDepsIterator.next().getConfiguration().getChecksum());
  }

  @Test
  public void testAlias() throws Exception {
    helper.useRuleClassProvider(setRuleClassProviders(getSimpleRule()).build());
    writeFile(
        "test/BUILD",
        "simple_rule(name = 'my_rule')",
        "alias(name = 'my_alias', actual = ':my_rule')");

    AnalysisProtos.ConfiguredTarget alias =
        Iterables.getOnlyElement(getOutput("//test:my_alias").getResultsList());

    assertThat(alias.getTarget().getRule().getName()).isEqualTo("//test:my_alias");
    assertThat(alias.getTarget().getRule().getRuleInputCount()).isEqualTo(1);
    assertThat(alias.getTarget().getRule().getRuleInput(0)).isEqualTo("//test:my_rule");
  }

  /* See b/209787345 for context. */
  @Test
  public void testAlias_withSelect() throws Exception {
    helper.useRuleClassProvider(setRuleClassProviders(getSimpleRule()).build());
    writeFile(
        "test/BUILD",
        "alias(",
        "  name = 'my_alias_rule',",
        "  actual = select({",
        "    ':config1': ':target1',",
        "    '//conditions:default': ':target2',",
        "  }),",
        ")",
        "config_setting(",
        "  name = 'config1',",
        "  values = {'test_arg': 'woof'},",
        ")",
        "simple_rule(name = 'target1')",
        "simple_rule(name = 'target2')");
    getHelper().useConfiguration("--test_arg=woof");
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);

    List<AnalysisProtos.ConfiguredTarget> myAliasRuleProto =
        getOutput("deps(//test:my_alias_rule)").getResultsList();

    List<String> depNames = new ArrayList<>(myAliasRuleProto.size());
    myAliasRuleProto.forEach(
        configuredTarget -> depNames.add(configuredTarget.getTarget().getRule().getName()));
    assertThat(depNames)
        .containsExactly("//test:my_alias_rule", "//test:config1", "//test:target1");
  }

  private MockRule getSimpleRule() {
    return () -> MockRule.define("simple_rule");
  }

  private AnalysisProtos.CqueryResult getOutput(String queryExpression) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<KeyedConfiguredTarget> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);

    ProtoOutputFormatterCallback callback =
        new ProtoOutputFormatterCallback(
            reporter,
            options,
            /*out=*/ null,
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            options.aspectDeps.createResolver(
                getHelper().getPackageManager(), NullEventHandler.INSTANCE),
            OutputType.BINARY);
    env.evaluateQuery(expression, callback);
    return callback.getProtoResult();
  }
}
