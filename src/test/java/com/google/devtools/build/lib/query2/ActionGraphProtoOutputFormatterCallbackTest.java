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
package com.google.devtools.build.lib.query2;

import static com.google.common.truth.Truth.assertThat;

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtos.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtos.KeyValuePair;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.query2.ActionGraphProtoOutputFormatterCallback.OutputType;
import com.google.devtools.build.lib.query2.engine.ActionGraphQueryHelper;
import com.google.devtools.build.lib.query2.engine.ActionGraphQueryTest;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.Setting;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.query2.output.AqueryOptions;
import com.google.devtools.build.lib.query2.output.AspectResolver.Mode;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetValue;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Pattern;
import org.junit.Before;
import org.junit.Test;

/** Tests for aquery's proto output format */
public class ActionGraphProtoOutputFormatterCallbackTest extends ActionGraphQueryTest {
  private AqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  @Before
  public final void setUpCqueryOptions() {
    this.options = new AqueryOptions();
    options.aspectDeps = Mode.OFF;
    this.reporter = new Reporter(new EventBus(), events::add);
  }

  @Test
  public void testBasicFunctionality() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'],",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    ActionGraphContainer actionGraphContainer = getOutput("//test:foo");
    Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());
    assertThat(action.getMnemonic()).isEqualTo("Genrule");
    String inputId = null;
    for (Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (artifact.getExecPath().equals("test/in")) {
        inputId = artifact.getId();
        break;
      }
    }
    assertThat(inputId).isNotNull();
    String inputDepSetId = Iterables.getOnlyElement(action.getInputDepSetIdsList());
    DepSetOfFiles depSetOfFiles =
        Iterables.getOnlyElement(actionGraphContainer.getDepSetOfFilesList());
    assertThat(depSetOfFiles.getId()).isEqualTo(inputDepSetId);
    assertThat(depSetOfFiles.getDirectArtifactIdsList()).contains(inputId);
    KeyValuePair infoItem = Iterables.getOnlyElement(action.getExecutionInfoList());
    assertThat(infoItem).isEqualTo(KeyValuePair.newBuilder().setKey("requires-x").build());
  }

  @Test
  public void testAqueryFilters_allFunctions_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='wrong_inputs', srcs=['wrong_in'], outs=['matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='wrong_outputs', srcs=['matching_in.java'], outs=['wrong_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='wrong_mnemonic', srcs=['in_bar.java'])");

    String inputs = ".*\\.java";
    String outputs = ".*matching_out";
    String mnemonic = "Genrule";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(
            ImmutableMap.of("inputs", inputs, "outputs", outputs, "mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(
            String.format(
                "inputs('%s', outputs('%s', mnemonic('%s', deps(//test:all))))",
                inputs, outputs, mnemonic),
            actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_regex_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String inputs = ".*matching_in.java";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("inputs", inputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("inputs('%s', deps(//test:all))", inputs), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testOutputsFilter_regex_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='wrong_outputs', srcs=['matching_in'], outs=['wrong_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')");

    String outputs = ".*matching_out";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("outputs", outputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("outputs('%s', deps(//test:all))", outputs), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testMnemonicFilter_regex_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String mnemonic = ".*rule";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("mnemonic('%s', deps(//test:all))", mnemonic), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_exactFileName_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String inputs = "test/foo_matching_in.java";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("inputs", inputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("inputs('%s', deps(//test:all))", inputs), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testOutputsFilter_exactFileName_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='wrong_outputs', srcs=['matching_in'], outs=['wrong_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')");

    String outputs = ".*/genfiles/test/foo_matching_out";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("outputs", outputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("outputs('%s', deps(//test:all))", outputs), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testMnemonicFilter_exactMnemonicValue_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String mnemonic = "Genrule";
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("mnemonic('%s', deps(//test:all))", mnemonic), actionFilters);

    assertMatchingOnlyFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("inputs", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("inputs('something', //test:foo)", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void testOutputsFilter_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("outputs", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("outputs('something', //test:foo)", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void testMnemonicFilter_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    ImmutableMap<String, Pattern> actionFilters =
        constructActionFilter(ImmutableMap.of("mnemonic", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("mnemonic('something', //test:foo)", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  private AnalysisProtos.ActionGraphContainer getOutput(String queryExpression) throws Exception {
    return getOutput(queryExpression, /* actionFilters= */ ImmutableMap.of());
  }

  private AnalysisProtos.ActionGraphContainer getOutput(
      String queryExpression, ImmutableMap<String, Pattern> actionFilters) throws Exception {
    QueryExpression expression = QueryParser.parse(queryExpression, getDefaultFunctions());
    Set<String> targetPatternSet = new LinkedHashSet<>();
    expression.collectTargetPatterns(targetPatternSet);
    helper.setQuerySettings(Setting.NO_IMPLICIT_DEPS);
    PostAnalysisQueryEnvironment<ConfiguredTargetValue> env =
        ((ActionGraphQueryHelper) helper).getPostAnalysisQueryEnvironment(targetPatternSet);

    ActionGraphProtoOutputFormatterCallback callback =
        new ActionGraphProtoOutputFormatterCallback(
            reporter,
            options,
            /*out=*/ null,
            getHelper().getSkyframeExecutor(),
            env.getAccessor(),
            OutputType.BINARY,
            actionFilters);
    env.evaluateQuery(expression, callback);
    return callback.getProtoResult();
  }

  private void assertMatchingOnlyFoo(ActionGraphContainer actionGraphContainer) {
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(1);
    Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());

    // Verify mnemonic
    assertThat(action.getMnemonic()).isEqualTo("Genrule");

    // Verify input
    String inputId = null;
    for (Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (artifact.getExecPath().equals("test/foo_matching_in.java")) {
        inputId = artifact.getId();
        break;
      }
    }
    assertThat(action.getInputDepSetIdsList()).contains(inputId);

    // Verify output
    String outputId = null;
    for (Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (artifact.getExecPath().endsWith("/genfiles/test/foo_matching_out")) {
        outputId = artifact.getId();
        break;
      }
    }
    assertThat(action.getOutputIdsList()).contains(outputId);
  }

  private ImmutableMap<String, Pattern> constructActionFilter(
      ImmutableMap<String, String> patternStrings) {
    ImmutableMap.Builder<String, Pattern> builder = ImmutableMap.builder();
    for (Entry<String, String> e : patternStrings.entrySet()) {
      builder.put(e.getKey(), Pattern.compile(e.getValue()));
    }
    return builder.build();
  }
}
