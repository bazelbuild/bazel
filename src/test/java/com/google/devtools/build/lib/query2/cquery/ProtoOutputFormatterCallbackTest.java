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
import static com.google.common.truth.extensions.proto.ProtoTruth.assertThat;
import static com.google.devtools.build.lib.packages.Attribute.attr;
import static com.google.devtools.build.lib.packages.BuildType.LABEL_LIST;
import static java.nio.charset.StandardCharsets.UTF_8;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Configuration;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.config.ExecutionTransitionFactory;
import com.google.devtools.build.lib.analysis.util.MockRule;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.NullEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.RuleClassProvider;
import com.google.devtools.build.lib.query2.PostAnalysisQueryEnvironment;
import com.google.devtools.build.lib.query2.cquery.CqueryOptions.Transitions;
import com.google.devtools.build.lib.query2.cquery.ProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.proto.proto2api.Build;
import com.google.devtools.build.lib.query2.proto.proto2api.Build.ConfiguredRuleInput;
import com.google.devtools.build.lib.query2.query.aspectresolvers.AspectResolver.Mode;
import com.google.devtools.build.lib.util.FileTypeSet;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.Message;
import com.google.protobuf.Parser;
import com.google.protobuf.TextFormat;
import com.google.protobuf.util.JsonFormat;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;
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
    ConfiguredRuleClassProvider ruleClassProvider = setRuleClassProviders(depsRule).build();
    helper.useRuleClassProvider(ruleClassProvider);

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
        "  values = {'foo': 'cat'}",
        ")");

    AnalysisProtosV2.ConfiguredTarget myRuleProto =
        Iterables.getOnlyElement(
            getProtoOutput(
                    "//test:my_rule", ruleClassProvider, AnalysisProtosV2.CqueryResult.parser())
                .getResultsList());
    List<Build.Attribute> attributes = myRuleProto.getTarget().getRule().getAttributeList();
    for (Build.Attribute attribute : attributes) {
      if (!attribute.getName().equals("deps")) {
        continue;
      }
      assertThat(attribute.getStringListValueCount()).isEqualTo(1);
      assertThat(attribute.getStringListValue(0)).isEqualTo("//test:mondays.java");
      break;
    }

    getHelper().useConfiguration("--foo=cat");
    myRuleProto =
        Iterables.getOnlyElement(
            getProtoOutput(
                    "//test:my_rule", ruleClassProvider, AnalysisProtosV2.CqueryResult.parser())
                .getResultsList());
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
  @SuppressWarnings("deprecation") // only use for tests
  public void testConfigurations() throws Exception {
    options.transitions = Transitions.LITE;

    MockRule ruleWithPatch =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder.add(
                        attr("deps", LABEL_LIST)
                            .allowedFileTypes(FileTypeSet.ANY_FILE)
                            .cfg(ExecutionTransitionFactory.createFactory())));
    MockRule parentRuleClass =
        () ->
            MockRule.define(
                "parent_rule",
                (builder, env) ->
                    builder.add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));

    ConfiguredRuleClassProvider ruleClassProvider =
        setRuleClassProviders(ruleWithPatch, parentRuleClass, getSimpleRule()).build();
    helper.useRuleClassProvider(ruleClassProvider);

    writeFile(
        "test/BUILD",
        "parent_rule(name = 'parent_rule',",
        "  deps = [':transition_rule'],",
        ")",
        "my_rule(name = 'transition_rule',",
        "  deps = [':patched', ':dep'],",
        ")",
        "simple_rule(name = 'dep')");

    AnalysisProtosV2.CqueryResult cqueryResult =
        getProtoOutput(
            "deps(//test:parent_rule)", ruleClassProvider, AnalysisProtosV2.CqueryResult.parser());
    List<Configuration> configurations = cqueryResult.getConfigurationsList();
    assertThat(configurations).hasSize(2);

    List<AnalysisProtosV2.ConfiguredTarget> resultsList = cqueryResult.getResultsList();

    AnalysisProtosV2.ConfiguredTarget parentRuleProto =
        getRuleProtoByName(resultsList, "//test:parent_rule");
    Set<ConfiguredTarget> keyedTargets = eval("deps(//test:parent_rule)");

    ConfiguredTarget parentRule = getKeyedTargetByLabel(keyedTargets, "//test:parent_rule");
    assertThat(parentRuleProto.getConfiguration().getChecksum())
        .isEqualTo(parentRule.getConfigurationChecksum());

    Configuration parentConfiguration =
        getConfigurationForId(configurations, parentRuleProto.getConfigurationId());
    assertThat(parentConfiguration.getChecksum()).isEqualTo(parentRule.getConfigurationChecksum());
    assertThat(parentConfiguration)
        .ignoringFieldDescriptors(
            Configuration.getDescriptor().findFieldByName("checksum"),
            Configuration.getDescriptor().findFieldByName("id"))
        .isEqualTo(
            Configuration.newBuilder()
                .setMnemonic("k8-fastbuild")
                .setPlatformName("k8")
                .setIsTool(false)
                .build());

    AnalysisProtosV2.ConfiguredTarget transitionRuleProto =
        getRuleProtoByName(resultsList, "//test:transition_rule");
    ConfiguredTarget transitionRule = getKeyedTargetByLabel(keyedTargets, "//test:transition_rule");
    assertThat(transitionRuleProto.getConfiguration().getChecksum())
        .isEqualTo(transitionRule.getConfigurationChecksum());

    Configuration transitionConfiguration =
        getConfigurationForId(configurations, transitionRuleProto.getConfigurationId());
    assertThat(transitionConfiguration.getChecksum())
        .isEqualTo(transitionRule.getConfigurationChecksum());

    AnalysisProtosV2.ConfiguredTarget depRuleProto = getRuleProtoByName(resultsList, "//test:dep");
    Configuration depRuleConfiguration =
        getConfigurationForId(configurations, depRuleProto.getConfigurationId());
    assertThat(depRuleConfiguration.getPlatformName()).isEqualTo("k8");
    assertThat(depRuleConfiguration.getMnemonic()).matches("k8-opt-exec-.*");
    assertThat(depRuleConfiguration.getIsTool()).isTrue();

    ConfiguredTarget depRule = getKeyedTargetByLabel(keyedTargets, "//test:dep");

    assertThat(depRuleProto.getConfiguration().getChecksum())
        .isEqualTo(depRule.getConfigurationChecksum());

    // Assert the proto checksums for targets in different configurations are not the same.
    assertThat(depRuleConfiguration.getChecksum())
        .isNotEqualTo(transitionConfiguration.getChecksum());
    // Targets without a configuration have a configuration_id of 0.
    AnalysisProtosV2.ConfiguredTarget fileTargetProto =
        resultsList.stream()
            .filter(result -> "//test:patched".equals(result.getTarget().getSourceFile().getName()))
            .findAny()
            .orElseThrow();
    assertThat(fileTargetProto.getConfigurationId()).isEqualTo(0);

    // Targets whose deps have no transitions should appear without configuration information.
    assertThat(parentRuleProto.getTarget().getRule().getConfiguredRuleInputList())
        .containsExactly(
            ConfiguredRuleInput.newBuilder().setLabel("//test:transition_rule").build());

    // Targets with deps with transitions should show them.
    ConfiguredRuleInput patchedConfiguredRuleInput =
        ConfiguredRuleInput.newBuilder().setLabel("//test:patched").build();
    ConfiguredRuleInput depConfiguredRuleInput =
        ConfiguredRuleInput.newBuilder()
            .setLabel("//test:dep")
            .setConfigurationChecksum(depRuleProto.getConfiguration().getChecksum())
            .setConfigurationId(depRuleProto.getConfigurationId())
            .build();
    List<ConfiguredRuleInput> configuredRuleInputs =
        transitionRuleProto.getTarget().getRule().getConfiguredRuleInputList();
    assertThat(configuredRuleInputs)
        .containsExactly(patchedConfiguredRuleInput, depConfiguredRuleInput);
  }

  private ConfiguredTarget getKeyedTargetByLabel(Set<ConfiguredTarget> keyedTargets, String label) {
    return Iterables.getOnlyElement(
        keyedTargets.stream()
            .filter(t -> label.equals(t.getLabel().getCanonicalForm()))
            .collect(Collectors.toSet()));
  }

  private Configuration getConfigurationForId(List<Configuration> configurations, int id) {
    return configurations.stream().filter(c -> c.getId() == id).findAny().orElseThrow();
  }

  private AnalysisProtosV2.ConfiguredTarget getRuleProtoByName(
      List<AnalysisProtosV2.ConfiguredTarget> resultsList, String s) {
    return resultsList.stream()
        .filter(result -> s.equals(result.getTarget().getRule().getName()))
        .findAny()
        .orElseThrow();
  }

  @Test
  public void testAlias() throws Exception {
    ConfiguredRuleClassProvider ruleClassProvider = setRuleClassProviders(getSimpleRule()).build();
    helper.useRuleClassProvider(ruleClassProvider);

    writeFile(
        "test/BUILD",
        "simple_rule(name = 'my_rule')",
        "alias(name = 'my_alias', actual = ':my_rule')");

    AnalysisProtosV2.ConfiguredTarget alias =
        Iterables.getOnlyElement(
            getProtoOutput(
                    "//test:my_alias", ruleClassProvider, AnalysisProtosV2.CqueryResult.parser())
                .getResultsList());

    assertThat(alias.getTarget().getRule().getName()).isEqualTo("//test:my_alias");
    assertThat(alias.getTarget().getRule().getRuleInputCount()).isEqualTo(1);
    assertThat(alias.getTarget().getRule().getRuleInput(0)).isEqualTo("//test:my_rule");
  }

  /* See b/209787345 for context. */
  @Test
  public void testAlias_withSelect() throws Exception {
    ConfiguredRuleClassProvider ruleClassProvider = setRuleClassProviders(getSimpleRule()).build();
    helper.useRuleClassProvider(ruleClassProvider);

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
        "  values = {'foo': 'woof'},",
        ")",
        "simple_rule(name = 'target1')",
        "simple_rule(name = 'target2')");
    getHelper().useConfiguration("--foo=woof");
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);

    List<AnalysisProtosV2.ConfiguredTarget> myAliasRuleProto =
        getProtoOutput(
                "deps(//test:my_alias_rule)",
                ruleClassProvider,
                AnalysisProtosV2.CqueryResult.parser())
            .getResultsList();

    List<String> depNames = new ArrayList<>(myAliasRuleProto.size());
    myAliasRuleProto.forEach(
        configuredTarget -> depNames.add(configuredTarget.getTarget().getRule().getName()));
    assertThat(depNames)
        // The alias also includes platform info since aliases with select() trigger toolchain
        // resolution. We're not interested in those here.
        .containsAtLeast("//test:my_alias_rule", "//test:config1", "//test:target1");
  }

  @Test
  public void testAllOutputFormatsEquivalentToProtoOutput() throws Exception {
    MockRule depsRule =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder.add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));
    ConfiguredRuleClassProvider ruleClassProvider = setRuleClassProviders(depsRule).build();
    helper.useRuleClassProvider(ruleClassProvider);

    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = ['lasagna.java', 'naps.java'],",
        ")");
    AnalysisProtosV2.CqueryResult prototype = AnalysisProtosV2.CqueryResult.getDefaultInstance();
    AnalysisProtosV2.CqueryResult protoOutput =
        getProtoOutput("//test:*", ruleClassProvider, prototype.getParserForType());

    AnalysisProtosV2.CqueryResult textprotoOutput =
        getProtoFromTextprotoOutput("//test:*", ruleClassProvider, prototype);

    AnalysisProtosV2.CqueryResult jsonprotoOutput =
        getProtoFromJsonprotoOutput("//test:*", ruleClassProvider, prototype);

    ImmutableList<AnalysisProtosV2.CqueryResult> streamedProtoOutput =
        getStreamedProtoOutput("//test:*", ruleClassProvider, prototype.getParserForType());
    AnalysisProtosV2.CqueryResult.Builder combinedStreamedProtoBuilder =
        AnalysisProtosV2.CqueryResult.newBuilder();
    for (AnalysisProtosV2.CqueryResult result : streamedProtoOutput) {
      if (!result.getResultsList().isEmpty()) {
        combinedStreamedProtoBuilder.addAllResults(result.getResultsList());
      }
      if (!result.getConfigurationsList().isEmpty()) {
        combinedStreamedProtoBuilder.addAllConfigurations(result.getConfigurationsList());
      }
    }

    assertThat(textprotoOutput).ignoringRepeatedFieldOrder().isEqualTo(protoOutput);
    assertThat(jsonprotoOutput).ignoringRepeatedFieldOrder().isEqualTo(protoOutput);
    assertThat(combinedStreamedProtoBuilder.build())
        .ignoringRepeatedFieldOrder()
        .isEqualTo(protoOutput);
  }

  @Test
  public void testAllOutputFormatsEquivalentToProtoOutput_noIncludeConfigurations()
      throws Exception {
    options.protoIncludeConfigurations = false;
    MockRule depsRule =
        () ->
            MockRule.define(
                "my_rule",
                (builder, env) ->
                    builder.add(attr("deps", LABEL_LIST).allowedFileTypes(FileTypeSet.ANY_FILE)));
    ConfiguredRuleClassProvider ruleClassProvider = setRuleClassProviders(depsRule).build();
    helper.useRuleClassProvider(ruleClassProvider);

    writeFile(
        "test/BUILD",
        "my_rule(",
        "  name = 'my_rule',",
        "  deps = ['lasagna.java', 'naps.java'],",
        ")");
    Build.QueryResult prototype = Build.QueryResult.getDefaultInstance();
    Build.QueryResult protoOutput =
        getProtoOutput("//test:*", ruleClassProvider, prototype.getParserForType());

    Build.QueryResult textprotoOutput =
        getProtoFromTextprotoOutput("//test:*", ruleClassProvider, prototype);

    Build.QueryResult jsonprotoOutput =
        getProtoFromJsonprotoOutput("//test:*", ruleClassProvider, prototype);

    ImmutableList<Build.QueryResult> streamedProtoOutput =
        getStreamedProtoOutput("//test:*", ruleClassProvider, prototype.getParserForType());
    Build.QueryResult.Builder combinedStreamedProtoBuilder = Build.QueryResult.newBuilder();
    for (Build.QueryResult result : streamedProtoOutput) {
      if (!result.getTargetList().isEmpty()) {
        combinedStreamedProtoBuilder.addAllTarget(result.getTargetList());
      }
    }

    assertThat(textprotoOutput).ignoringRepeatedFieldOrder().isEqualTo(protoOutput);
    assertThat(jsonprotoOutput).ignoringRepeatedFieldOrder().isEqualTo(protoOutput);
    assertThat(combinedStreamedProtoBuilder.build())
        .ignoringRepeatedFieldOrder()
        .isEqualTo(protoOutput);
  }

  private MockRule getSimpleRule() {
    return () -> MockRule.define("simple_rule");
  }

  private <T extends Message> T getProtoOutput(
      String queryExpression, RuleClassProvider ruleClassProvider, Parser<T> parser)
      throws Exception {
    InputStream in = queryAndGetInputStream(queryExpression, ruleClassProvider, OutputType.BINARY);
    return parser.parseFrom(in, ExtensionRegistry.getEmptyRegistry());
  }

  private <T extends Message> ImmutableList<T> getStreamedProtoOutput(
      String queryExpression, RuleClassProvider ruleClassProvider, Parser<T> parser)
      throws Exception {
    InputStream in =
        queryAndGetInputStream(queryExpression, ruleClassProvider, OutputType.DELIMITED_BINARY);
    ImmutableList.Builder<T> builder = new ImmutableList.Builder<>();
    T result;
    while ((result = parser.parseDelimitedFrom(in, ExtensionRegistry.getEmptyRegistry())) != null) {
      builder.add(result);
    }
    return builder.build();
  }

  private <T extends Message> T getProtoFromTextprotoOutput(
      String queryExpression, RuleClassProvider ruleClassProvider, T prototype) throws Exception {
    InputStream in = queryAndGetInputStream(queryExpression, ruleClassProvider, OutputType.TEXT);
    Message.Builder builder = prototype.newBuilderForType();
    TextFormat.getParser().merge(new InputStreamReader(in, UTF_8), builder);
    @SuppressWarnings("unchecked")
    T message = (T) builder.build();
    return message;
  }

  private <T extends Message> T getProtoFromJsonprotoOutput(
      String queryExpression, RuleClassProvider ruleClassProvider, T prototype) throws Exception {
    InputStream in = queryAndGetInputStream(queryExpression, ruleClassProvider, OutputType.JSON);
    Message.Builder builder = prototype.newBuilderForType();
    JsonFormat.parser().merge(new InputStreamReader(in, UTF_8), builder);
    @SuppressWarnings("unchecked")
    T message = (T) builder.build();
    return message;
  }

  private InputStream queryAndGetInputStream(
      String queryExpression, RuleClassProvider ruleClassProvider, OutputType outputType)
      throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<ConfiguredTarget> env =
        ((ConfiguredTargetQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    ProtoOutputFormatterCallback callback =
        new ProtoOutputFormatterCallback(
            reporter,
            options,
            out,
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            options.aspectDeps.createResolver(
                getHelper().getPackageManager(), NullEventHandler.INSTANCE),
            outputType,
            ruleClassProvider);
    env.evaluateQuery(expression, callback);
    return new ByteArrayInputStream(out.toByteArray());
  }
}
