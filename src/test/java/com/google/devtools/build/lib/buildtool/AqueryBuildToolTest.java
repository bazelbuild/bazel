// Copyright 2020 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.buildtool;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.truth.Truth.assertThat;
import static java.nio.charset.StandardCharsets.UTF_8;
import static org.junit.Assert.assertThrows;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Action;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.ActionGraphContainer;
import com.google.devtools.build.lib.buildtool.AqueryProcessor.AqueryActionFilterException;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.TargetPattern;
import com.google.devtools.build.lib.events.EventKind;
import com.google.devtools.build.lib.query2.aquery.ActionGraphQueryEnvironment;
import com.google.devtools.build.lib.query2.aquery.AqueryOptions;
import com.google.devtools.build.lib.query2.engine.QueryEnvironment.QueryFunction;
import com.google.devtools.build.lib.query2.engine.QueryExpression;
import com.google.devtools.build.lib.query2.engine.QueryParser;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.runtime.commands.AqueryCommand;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery.Code;
import com.google.protobuf.ExtensionRegistry;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Integration tests for aquery. */
@RunWith(JUnit4.class)
public class AqueryBuildToolTest extends BuildIntegrationTestCase {
  private ImmutableMap<String, QueryFunction> functions;

  @Before
  public final void setFunctions() {
    ImmutableMap.Builder<String, QueryFunction> builder = ImmutableMap.builder();

    for (QueryFunction queryFunction : ActionGraphQueryEnvironment.FUNCTIONS) {
      builder.put(queryFunction.getName(), queryFunction);
    }

    for (QueryFunction queryFunction : ActionGraphQueryEnvironment.AQUERY_FUNCTIONS) {
      builder.put(queryFunction.getName(), queryFunction);
    }

    functions = builder.buildOrThrow();
    runtimeWrapper.addOptionsClass(AqueryOptions.class);
  }

  @Test
  public void testConstructor_wrongAqueryFilterFormat_throwsError() throws Exception {
    QueryExpression expr = QueryParser.parse("deps(inputs('abc', //abc))", functions);

    assertThrows(
        AqueryActionFilterException.class,
        () -> new AqueryProcessor(expr, TargetPattern.defaultParser()));
  }

  @Test
  public void testConstructor_wrongPatternSyntax_throwsError() throws Exception {
    QueryExpression expr = QueryParser.parse("inputs('*abc', //abc)", functions);

    AqueryActionFilterException thrown =
        assertThrows(
            AqueryActionFilterException.class,
            () -> new AqueryProcessor(expr, TargetPattern.defaultParser()));
    assertThat(thrown).hasMessageThat().contains("Wrong query syntax:");
  }

  @Test
  public void testDmpActionGraphFromSkyframe_wrongOutputFormat_returnsFailure() throws Exception {
    addOptions("--output=text");
    CommandEnvironment env = runtimeWrapper.newCommand(AqueryCommand.class);
    AqueryProcessor aqueryProcessor = new AqueryProcessor(null, TargetPattern.defaultParser());
    BlazeCommandResult result = aqueryProcessor.dumpActionGraphFromSkyframe(env);

    assertThat(result.isSuccess()).isFalse();
    assertThat(result.getDetailedExitCode().getFailureDetail().getActionQuery().getCode())
        .isEqualTo(Code.SKYFRAME_STATE_PREREQ_UNMET);
  }

  @Test
  public void testAquerySkyframeStateProtoNotCutoff() throws Exception {
    // First, prepare and run the build.
    write(
        "x/BUILD",
        """
        genrule(
            name = "x",
            srcs = ["in"],
            # This has the length 10, so it will include a 0x0a / newline character
            # that triggers the cutoff.
            outs = ["1234567890"],
            cmd = "touch $(OUTS)",
        )
        """);
    write("x/in", "");
    buildTarget("//x");

    // Then, run aquery and dump the action graph as of the previous skyframe state.
    addOptions("--output=proto", "--skyframe_state");
    CommandEnvironment env = runtimeWrapper.newCommand(AqueryCommand.class);
    ByteArrayOutputStream stdout = captureStdout(env);

    AqueryProcessor aqueryProcessor = new AqueryProcessor(null, TargetPattern.defaultParser());
    BlazeCommandResult result = aqueryProcessor.dumpActionGraphFromSkyframe(env);
    assertThat(result.isSuccess()).isTrue();

    // Test whether stdout is a valid proto.
    assertThat(stdout.size()).isGreaterThan(0);
    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    assertThat(actionGraphContainer.getActionsList()).isNotEmpty();
  }

  @Test
  public void testAqueryProgressMessage() throws Exception {
    write(
        "x/BUILD",
        """
        genrule(
            name = "x",
            srcs = ["in"],
            outs = ["out"],
            cmd = "touch $(OUTS)",
        )
        """);
    write("x/in", "");
    buildTarget("//x");

    addOptions("--output=proto", "--skyframe_state");
    CommandEnvironment env = runtimeWrapper.newCommand(AqueryCommand.class);
    ByteArrayOutputStream stdout = captureStdout(env);

    AqueryProcessor aqueryProcessor = new AqueryProcessor(null, TargetPattern.defaultParser());
    BlazeCommandResult result = aqueryProcessor.dumpActionGraphFromSkyframe(env);
    assertThat(result.isSuccess()).isTrue();

    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    Action genruleAction =
        actionGraphContainer.getActionsList().stream()
            .filter(action -> action.getMnemonic().equals("Genrule"))
            .findFirst()
            .orElseThrow(() -> new AssertionError("No Genrule action found in the action graph."));

    assertThat(genruleAction.getProgressMessage()).contains("Executing genrule //x:x");
  }

  private void setupBrokenPackage() throws Exception {
    write(
        "pkg/BUILD",
        """
        genrule(
            name = "good",
            srcs = ["in.txt"],
            outs = ["out_good.txt"],
            cmd = "touch $(OUTS)",
        )
        genrule(
            name = "bad",
            srcs = ["//nonexistent:missing"],
            outs = ["out_bad.txt"],
            cmd = "touch $(OUTS)",
        )
        """);
    write("pkg/in.txt", "");
  }

  @Test
  public void testAqueryProto_keepGoing_withBrokenTarget() throws Exception {
    setupBrokenPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=proto", "--keep_going");
    assertThrows(
        BuildFailedException.class, () -> runtimeWrapper.runAqueryExprCommand("//pkg:all"));

    assertThat(stdout.size()).isGreaterThan(0);
    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    assertThat(actionGraphContainer.getActionsList()).isNotEmpty();
  }

  @Test
  public void testAquerySummary_keepGoing_withBrokenTarget() throws Exception {
    setupBrokenPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=summary", "--keep_going");
    assertThrows(
        BuildFailedException.class, () -> runtimeWrapper.runAqueryExprCommand("//pkg:all"));

    assertThat(stdout.toString(UTF_8)).contains("Genrule");
  }

  private ByteArrayOutputStream captureReporterStdout() {
    ByteArrayOutputStream stdout = new ByteArrayOutputStream();
    events.addHandler(
        event -> {
          if (event.getKind().equals(EventKind.STDOUT)) {
            try {
              stdout.write(event.getMessageBytes());
            } catch (IOException e) {
              throw new IllegalStateException(e);
            }
          }
        });
    return stdout;
  }

  private ByteArrayOutputStream captureStdout(CommandEnvironment env) {
    ByteArrayOutputStream stdout = new ByteArrayOutputStream();
    env.getReporter()
        .addHandler(
            event -> {
              if (event.getKind().equals(EventKind.STDOUT)) {
                try {
                  stdout.write(event.getMessageBytes());
                } catch (IOException e) {
                  throw new IllegalStateException(e);
                }
              }
            });
    return stdout;
  }

  private void setupTwoOutputsPackage() throws Exception {
    write(
        "test/defs.bzl",
        """
        def _two_outputs_impl(ctx):
            out1 = ctx.actions.declare_file(ctx.label.name + ".out1")
            out2 = ctx.actions.declare_file(ctx.label.name + ".out2")
            ctx.actions.run_shell(
                outputs = [out1],
                command = "echo 1 > " + out1.path,
                mnemonic = "ActionOne",
            )
            ctx.actions.run_shell(
                outputs = [out2],
                command = "echo 2 > " + out2.path,
                mnemonic = "ActionTwo",
            )
            return [
                DefaultInfo(files = depset([out1])),
                OutputGroupInfo(extra = depset([out2])),
            ]

        two_outputs = rule(implementation = _two_outputs_impl)
        """);
    write(
        "test/BUILD",
        """
        load(":defs.bzl", "two_outputs")
        two_outputs(name = "my_target")
        """);
  }

  @Test
  public void testPruneUnusedActionsFalse_emitsAllActions() throws Exception {
    setupTwoOutputsPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=proto", "--noprune_unused_actions");
    runtimeWrapper.runAqueryExprCommand("//test:my_target");

    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    ImmutableList<String> mnemonics =
        actionGraphContainer.getActionsList().stream()
            .map(Action::getMnemonic)
            .collect(toImmutableList());
    assertThat(mnemonics).containsExactly("ActionOne", "ActionTwo");
  }

  @Test
  public void testPruneUnusedActionsTrue_prunesUnusedActions() throws Exception {
    setupTwoOutputsPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=proto", "--prune_unused_actions");
    runtimeWrapper.runAqueryExprCommand("//test:my_target");

    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    ImmutableList<String> mnemonics =
        actionGraphContainer.getActionsList().stream()
            .map(Action::getMnemonic)
            .collect(toImmutableList());
    assertThat(mnemonics).containsExactly("ActionOne");
  }

  @Test
  public void testPruneUnusedActionsTrue_withOutputGroups_prunesActionsNotInOutputGroup()
      throws Exception {
    setupTwoOutputsPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=proto", "--prune_unused_actions", "--output_groups=extra");
    runtimeWrapper.runAqueryExprCommand("//test:my_target");

    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    ImmutableList<String> mnemonics =
        actionGraphContainer.getActionsList().stream()
            .map(Action::getMnemonic)
            .collect(toImmutableList());
    assertThat(mnemonics).containsExactly("ActionTwo");
  }

  @Test
  public void testPruneUnusedActionsTrue_textOutput_prunesUnusedActions() throws Exception {
    setupTwoOutputsPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=text", "--prune_unused_actions");
    runtimeWrapper.runAqueryExprCommand("//test:my_target");

    String outputText = stdout.toString(UTF_8);
    assertThat(outputText).contains("ActionOne");
    assertThat(outputText).doesNotContain("ActionTwo");
  }

  @Test
  public void testPruneUnusedActionsTrue_summaryOutput_prunesUnusedActions() throws Exception {
    setupTwoOutputsPackage();
    ByteArrayOutputStream stdout = captureReporterStdout();

    addOptions("--output=summary", "--prune_unused_actions");
    runtimeWrapper.runAqueryExprCommand("//test:my_target");

    String outputText = stdout.toString(UTF_8);
    assertThat(outputText).contains("ActionOne");
    assertThat(outputText).doesNotContain("ActionTwo");
  }

  @Test
  public void testPruneUnusedActions_depsQuery_prunesTransitiveUnusedActions() throws Exception {
    write(
        "deps_test/defs.bzl",
        """
        def _dep_rule_impl(ctx):
            out1 = ctx.actions.declare_file(ctx.label.name + ".out1")
            out2 = ctx.actions.declare_file(ctx.label.name + ".out2")
            ctx.actions.run_shell(
                outputs = [out1],
                command = "echo 1 > " + out1.path,
                mnemonic = "DepUsedAction",
            )
            ctx.actions.run_shell(
                outputs = [out2],
                command = "echo 2 > " + out2.path,
                mnemonic = "DepUnusedAction",
            )
            return [DefaultInfo(files = depset([out1]))]

        dep_rule = rule(implementation = _dep_rule_impl)

        def _top_rule_impl(ctx):
            out = ctx.actions.declare_file(ctx.label.name + ".out")
            inputs = ctx.files.deps
            ctx.actions.run_shell(
                inputs = inputs,
                outputs = [out],
                command = "cat " + " ".join([f.path for f in inputs]) + " > " + out.path,
                mnemonic = "TopAction",
            )
            return [DefaultInfo(files = depset([out]))]

        top_rule = rule(
            implementation = _top_rule_impl,
            attrs = {"deps": attr.label_list()},
        )
        """);
    write(
        "deps_test/BUILD",
        """
        load(":defs.bzl", "dep_rule", "top_rule")
        dep_rule(name = "dep_target")
        top_rule(name = "top_target", deps = [":dep_target"])
        """);

    ByteArrayOutputStream stdout = captureReporterStdout();
    addOptions("--output=proto", "--prune_unused_actions");
    runtimeWrapper.runAqueryExprCommand("deps(//deps_test:top_target)", "//deps_test:top_target");

    ActionGraphContainer actionGraphContainer =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    ImmutableList<String> mnemonics =
        actionGraphContainer.getActionsList().stream()
            .map(Action::getMnemonic)
            .collect(toImmutableList());
    assertThat(mnemonics).containsExactly("TopAction", "DepUsedAction");
  }

  @Test
  public void testDumpActionGraphFromSkyframe_pruneUnusedActionsTrue_returnsFailure()
      throws Exception {
    addOptions("--output=proto", "--prune_unused_actions", "--skyframe_state");
    CommandEnvironment env = runtimeWrapper.newCommand(AqueryCommand.class);
    AqueryProcessor aqueryProcessor = new AqueryProcessor(null, TargetPattern.defaultParser());
    BlazeCommandResult result = aqueryProcessor.dumpActionGraphFromSkyframe(env);

    assertThat(result.isSuccess()).isFalse();
    assertThat(result.getDetailedExitCode().getFailureDetail().getActionQuery().getCode())
        .isEqualTo(Code.SKYFRAME_STATE_PREREQ_UNMET);
  }
}
