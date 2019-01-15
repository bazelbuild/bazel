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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.devtools.build.lib.analysis.AnalysisProtos;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ActionGraphContainer;
import com.google.devtools.build.lib.analysis.AnalysisProtos.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtos.DepSetOfFiles;
import com.google.devtools.build.lib.analysis.AnalysisProtos.KeyValuePair;
import com.google.devtools.build.lib.analysis.AnalysisProtos.ParamFile;
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
import com.google.devtools.build.lib.util.OS;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map.Entry;
import java.util.Set;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.junit.Before;
import org.junit.Test;

/** Tests for aquery's proto output format */
public class ActionGraphProtoOutputFormatterCallbackTest extends ActionGraphQueryTest {
  private AqueryOptions options;
  private Reporter reporter;
  private final List<Event> events = new ArrayList<>();

  @Before
  public final void setUpAqueryOptions() {
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
    AqueryActionFilter actionFilters =
        constructActionFilter(
            ImmutableMap.of("inputs", inputs, "outputs", outputs, "mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(
            String.format(
                "inputs('%s', outputs('%s', mnemonic('%s', deps(//test:all))))",
                inputs, outputs, mnemonic),
            actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_regex_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String inputs = ".*matching_in.java";
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("inputs", inputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("inputs('%s', deps(//test:all))", inputs), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
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
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("outputs", outputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("outputs('%s', deps(//test:all))", outputs), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testMnemonicFilter_regex_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String mnemonic = ".*rule";
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("mnemonic('%s', deps(//test:all))", mnemonic), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_exactFileName_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String inputs = "test/foo_matching_in.java";
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("inputs", inputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("inputs('%s', deps(//test:all))", inputs), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
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
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("outputs", outputs));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("outputs('%s', deps(//test:all))", outputs), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testMnemonicFilter_exactMnemonicValue_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])");

    String mnemonic = "Genrule";
    AqueryActionFilter actionFilters = constructActionFilter(ImmutableMap.of("mnemonic", mnemonic));

    ActionGraphContainer actionGraphContainer =
        getOutput(String.format("mnemonic('%s', deps(//test:all))", mnemonic), actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testInputsFilter_chainInputs_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("inputs", ImmutableList.of("in", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("inputs('in', inputs('something', //test:foo))", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void testOutputsFilter_chainOutputs_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("outputs", ImmutableList.of("out", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("outputs('out', outputs('something', //test:foo))", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void testMnemonicFilter_chainMnemonics_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("mnemonic", ImmutableList.of(".*rule", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("mnemonic('.*rule', mnemonic('something', //test:foo))", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void testInputsFilter_chainInputs_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "java_library(name='bar', srcs=['in_bar.java'])",
        "genrule(name='foo2', srcs=['foo_matching_in.java_not'], outs=['foo_matching_out2'],",
        "        cmd='cat $(SRCS) > $(OUTS)')");

    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("inputs", ImmutableList.of(".*java", ".*foo.*"));

    ActionGraphContainer actionGraphContainer =
        getOutput("inputs('.*java', inputs('.*foo.*', deps(//test:all)))", actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testOutputsFilter_chainOutputs_matchingOnlyFooAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['foo_matching_in.java'], outs=['foo_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='foo2', srcs=['foo_matching_in.java'], outs=['foo_matching_out_not'],",
        "        cmd='cat $(SRCS) > $(OUTS)')",
        "genrule(name='foo3', srcs=['foo_matching_in.java'], outs=['not_matching_out'],",
        "        cmd='cat $(SRCS) > $(OUTS)')");

    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("outputs", ImmutableList.of(".*out", ".*foo.*"));

    ActionGraphContainer actionGraphContainer =
        getOutput("outputs('.*out', outputs('.*foo.*', deps(//test:all)))", actionFilters);

    assertMatchingOnlyActionFromFoo(actionGraphContainer);
  }

  @Test
  public void testMnemonicFilter_chainMnemonic_matchingOnlyFooAction() throws Exception {
    // java_library targets generate actions of the following mnemonics:
    // - Javac
    // - JavaSourceJar
    // - Turbine
    writeFile("test/BUILD", "java_library(", "    name = 'foo',", "    srcs = ['Foo.java'],", ")");

    AqueryActionFilter actionFilters =
        constructActionFilterChainSameFunction("mnemonic", ImmutableList.of("Java.*", ".*e.*"));

    ActionGraphContainer actionGraphContainer =
        getOutput("mnemonic('Java.*', mnemonic('.*e.*', deps(//test:all)))", actionFilters);

    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(1);
    Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());
    assertThat(action.getMnemonic()).isEqualTo("JavaSourceJar");
  }

  @Test
  public void testInputsFilter_noMatchingAction() throws Exception {
    writeFile(
        "test/BUILD",
        "genrule(name='foo', srcs=['in'], outs=['out'], tags=['requires-x'], ",
        "        cmd='cat $(SRCS) > $(OUTS)')");
    AqueryActionFilter actionFilters =
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
    AqueryActionFilter actionFilters =
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
    AqueryActionFilter actionFilters =
        constructActionFilter(ImmutableMap.of("mnemonic", "something"));
    ActionGraphContainer actionGraphContainer =
        getOutput("mnemonic('something', //test:foo)", actionFilters);
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(0);
  }

  @Test
  public void test_includeParamFile_subsetOfCmdlineArgs() throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      return;
    }
    options.includeParamFiles = true;
    options.includeCommandline = true;

    writeFile("test/BUILD", "cc_binary(name='foo', srcs=['foo.cc'])");
    ActionGraphContainer actionGraphContainer =
        getOutput("deps(//test:foo)", AqueryActionFilter.emptyInstance());
    Action cppLinkAction =
        Iterables.getOnlyElement(
            actionGraphContainer.getActionsList().stream()
                .filter(x -> x.getMnemonic().equals("CppLink"))
                .collect(Collectors.toList()));

    // Verify that there's exactly 1 param file.
    assertThat(cppLinkAction.getParamFilesCount()).isEqualTo(1);

    // Verify that the set of arguments in the param file
    // is a subset of the set of arguments in the command line.
    ParamFile paramFile = Iterables.getOnlyElement(cppLinkAction.getParamFilesList());
    Set<String> cmdlineArgs = new HashSet<>(cppLinkAction.getArgumentsList());
    Set<String> paramFileArgs = new HashSet<>(paramFile.getArgumentsList());
    assertThat(cmdlineArgs).containsAllIn(paramFileArgs);
  }

  @Test
  public void test_flagTurnedOff_excludeParamFile() throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      return;
    }
    writeFile("test/BUILD", "cc_binary(name='foo', srcs=['foo.cc'])");
    ActionGraphContainer actionGraphContainer =
        getOutput("deps(//test:foo)", AqueryActionFilter.emptyInstance());
    Action cppLinkAction =
        Iterables.getOnlyElement(
            actionGraphContainer.getActionsList().stream()
                .filter(x -> x.getMnemonic().equals("CppLink"))
                .collect(Collectors.toList()));

    // Verify that there's no param file field.
    assertThat(cppLinkAction.getParamFilesCount()).isEqualTo(0);
  }

  @Test
  public void test_includeParamFileSpawnActionStarlarkRule_noParamFileExplicitlyWritten()
      throws Exception {
    if (OS.getCurrent() == OS.DARWIN) {
      return;
    }
    options.includeParamFiles = true;
    options.includeCommandline = true;

    writeFile(
        "test/test_rule.bzl",
        "def _impl(ctx):",
        "  args = ctx.actions.args()",
        "  args.add('--param_file_arg')",
        "  args.set_param_file_format('multiline')",
        "  args.use_param_file('--param_file=%s', use_always = True)",
        "  ctx.actions.run(",
        "    inputs = ctx.files.srcs,",
        "    outputs = [ctx.outputs.outfile],",
        "    executable = 'dummy',",
        "    arguments = ['--non-param-file-flag', args],",
        "    mnemonic = 'SkylarkAction'",
        "  )",
        "test_rule = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True)",
        "  },",
        "  outputs = {",
        "    'outfile': '{name}.out'",
        "  },",
        ")");

    writeFile(
        "test/BUILD",
        "load('//test:test_rule.bzl', 'test_rule')",
        "test_rule(name='foo', srcs=['foo.java'])");

    ActionGraphContainer actionGraphContainer =
        getOutput("deps(//test:all)", AqueryActionFilter.emptyInstance());

    Action spawnAction = Iterables.getOnlyElement(actionGraphContainer.getActionsList());

    // Verify that there's no param file field.
    assertThat(spawnAction.getParamFilesCount()).isEqualTo(0);

    // Verify that the argument list contains both arguments from param file and otherwise.
    assertThat(spawnAction.getArgumentsList())
        .containsExactly("dummy", "--non-param-file-flag", "--param_file_arg");
  }

  @Test
  public void testCppActionTemplate_includesActionTemplateMnemonic() throws Exception {
    writeFile(
        "test/a.bzl",
        "def _impl(ctx):",
        "  directory = ctx.actions.declare_directory(ctx.attr.name + \"_artifact.cc\")",
        "  ctx.action(",
        "    inputs = ctx.files.srcs,",
        "    outputs = [directory],",
        "    mnemonic = 'MoveTreeArtifact',",
        "    command = 'echo abc'",
        "  )",
        "  return [DefaultInfo(files = depset([directory]))]",
        "cc_tree_artifact_files = rule(",
        "  implementation = _impl,",
        "  attrs = {",
        "    'srcs': attr.label_list(allow_files=True),",
        "  },",
        ")");

    writeFile(
        "test/BUILD",
        "load(':a.bzl', 'cc_tree_artifact_files')",
        "cc_tree_artifact_files(",
        "    name = 'tree_artifact',",
        "    srcs = ['a1.cc', 'a2.cc'],",
        ")",
        "",
        "cc_binary(",
        "    name = 'bin',",
        "    srcs = ['b1.h', 'b2.cc', ':tree_artifact'],",
        ")");

    ActionGraphContainer actionGraphContainer =
        getOutput("deps(//test:all)", AqueryActionFilter.emptyInstance());

    List<Action> cppCompileActionTemplates =
        actionGraphContainer.getActionsList().stream()
            .filter(action -> action.getMnemonic().equals("CppCompileActionTemplate"))
            .collect(Collectors.toList());

    // Darwin and Windows only produce 1 CppCompileActionTemplate with PIC,
    // while Linux has both PIC and non-PIC CppCompileActionTemplates
    int expectedActionsCount =
        (OS.getCurrent() == OS.DARWIN || OS.getCurrent() == OS.WINDOWS) ? 1 : 2;

    // Verify that we have the appropriate number of CppCompileActionTemplates.
    assertThat(cppCompileActionTemplates).hasSize(expectedActionsCount);
  }

  private AnalysisProtos.ActionGraphContainer getOutput(String queryExpression) throws Exception {
    return getOutput(queryExpression, /* actionFilters= */ AqueryActionFilter.emptyInstance());
  }

  private AnalysisProtos.ActionGraphContainer getOutput(
      String queryExpression, AqueryActionFilter actionFilters) throws Exception {
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

  private void assertMatchingOnlyActionFromFoo(ActionGraphContainer actionGraphContainer) {
    assertMatchingOnlyAction(
        actionGraphContainer,
        "Genrule",
        "test/foo_matching_in.java",
        "/genfiles/test/foo_matching_out");
  }

  private void assertMatchingOnlyAction(
      ActionGraphContainer actionGraphContainer,
      String mnemonic,
      String onlyInput,
      String onlyOutput) {
    assertThat(actionGraphContainer.getActionsCount()).isEqualTo(1);
    Action action = Iterables.getOnlyElement(actionGraphContainer.getActionsList());

    // Verify mnemonic
    assertThat(action.getMnemonic()).isEqualTo(mnemonic);

    // Verify input
    String inputId = null;
    for (Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (artifact.getExecPath().endsWith(onlyInput)) {
        inputId = artifact.getId();
        break;
      }
    }
    assertThat(action.getInputDepSetIdsList()).contains(inputId);

    // Verify output
    String outputId = null;
    for (Artifact artifact : actionGraphContainer.getArtifactsList()) {
      if (artifact.getExecPath().endsWith(onlyOutput)) {
        outputId = artifact.getId();
        break;
      }
    }
    assertThat(action.getOutputIdsList()).contains(outputId);
  }

  private AqueryActionFilter constructActionFilter(ImmutableMap<String, String> patternStrings) {
    AqueryActionFilter.Builder builder = AqueryActionFilter.builder();
    for (Entry<String, String> e : patternStrings.entrySet()) {
      builder.put(e.getKey(), Pattern.compile(e.getValue()));
    }
    return builder.build();
  }

  private AqueryActionFilter constructActionFilterChainSameFunction(
      String function, List<String> patternStrings) {
    AqueryActionFilter.Builder builder = AqueryActionFilter.builder();
    for (String s : patternStrings) {
      builder.put(function, Pattern.compile(s));
    }
    return builder.build();
  }
}
