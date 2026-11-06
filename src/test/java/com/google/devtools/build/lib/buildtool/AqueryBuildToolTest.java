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
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.Artifact;
import com.google.devtools.build.lib.analysis.AnalysisProtosV2.PathFragment;
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
import com.google.devtools.build.lib.util.OS;
import com.google.protobuf.ExtensionRegistry;
import com.google.protobuf.TextFormat;
import com.google.protobuf.util.JsonFormat;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
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

  private void setupMapDirectory() throws Exception {
    // Use the same Windows shell as BuildIntegrationTestCase: Windows cannot execute a shebang.
    String shell = OS.getCurrent() == OS.WINDOWS ? "c:/msys64/usr/bin/bash.exe" : "/bin/bash";
    String shellName = getFileSystem().getPath(shell).getBaseName();
    write(
        "mapped/defs.bzl",
        """
        def _expand(template_ctx, input_directories, output_directories, tools, **kwargs):
            for source in input_directories["sources"].children:
                first = template_ctx.declare_file(
                    source.basename, directory = output_directories["first"])
                second = template_ctx.declare_file(
                    source.basename, directory = output_directories["second"])
                args = template_ctx.args()
                args.add_all([tools["copy"], source, first, second])
                template_ctx.run(
                    executable = tools["shell"],
                    inputs = [source, tools["copy"]],
                    outputs = [first, second],
                    arguments = [args],
                )

        def _outputs(ctx, suffix):
            prefix = ctx.label.name + suffix
            sources = ctx.actions.declare_directory(prefix + ".sources")
            first = ctx.actions.declare_directory(prefix + ".first")
            second = ctx.actions.declare_directory(prefix + ".second")
            tool = ctx.actions.declare_file(prefix + ".copy.sh")
            ctx.actions.write(tool, '#!/bin/sh\\ncp "$1" "$2"; cp "$1" "$3"\\n',
                              is_executable = True)
            ctx.actions.run_shell(
                outputs = [sources],
                command = "mkdir -p {0}; echo a > {0}/a; echo b > {0}/b".format(sources.path),
            )
            ctx.actions.map_directory(
                input_directories = {"sources": sources},
                output_directories = {"first": first, "second": second},
                tools = {"copy": tool, "shell": ctx.file._shell},
                use_default_shell_env = True,
                implementation = _expand,
                mnemonic = "ExpandedCopy",
            )
            return [first, second]

        def _impl(ctx):
            return [DefaultInfo(files = depset(_outputs(ctx, "")))]

        def _aspect_impl(target, ctx):
            return [OutputGroupInfo(mapped = depset(_outputs(ctx, ".aspect")))]

        _attrs = {"_shell": attr.label(default = "//mapped:shell/%s", allow_single_file = True)}
        mapped_rule = rule(implementation = _impl, attrs = _attrs)
        mapped_aspect = aspect(implementation = _aspect_impl, attrs = _attrs)
        """
            .formatted(shellName));
    write(
        "mapped/BUILD",
        """
        load(":defs.bzl", "mapped_rule")
        mapped_rule(name = "example")
        """);
    // Keep the shell beside its native DLLs on Windows; only the executable is an action input.
    getWorkspace()
        .getRelative("mapped/shell")
        .createSymbolicLink(getFileSystem().getPath(shell).getParentDirectory());
  }

  private ActionGraphContainer dumpSkyframe(String format, String expression) throws Exception {
    addOptions("--output=" + format, "--skyframe_state");
    CommandEnvironment env = runtimeWrapper.newCommand(AqueryCommand.class);
    ByteArrayOutputStream stdout = captureStdout(env);
    QueryExpression query = expression == null ? null : QueryParser.parse(expression, functions);
    AqueryProcessor processor = new AqueryProcessor(query, TargetPattern.defaultParser());
    assertThat(processor.dumpActionGraphFromSkyframe(env).isSuccess()).isTrue();
    if (format.equals("proto")) {
      return ActionGraphContainer.parseFrom(
          stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    }
    ActionGraphContainer.Builder graph = ActionGraphContainer.newBuilder();
    if (format.equals("jsonproto")) {
      JsonFormat.parser().merge(stdout.toString(UTF_8), graph);
    } else {
      TextFormat.getParser().merge(stdout.toString(UTF_8), graph);
    }
    return graph.build();
  }

  private static String artifactPath(ActionGraphContainer graph, Artifact artifact) {
    Map<Integer, PathFragment> fragments = new HashMap<>();
    for (PathFragment fragment : graph.getPathFragmentsList()) {
      fragments.put(fragment.getId(), fragment);
    }
    String path = "";
    for (int id = artifact.getPathFragmentId(); id != 0; ) {
      PathFragment fragment = fragments.get(id);
      path = fragment.getLabel() + (path.isEmpty() ? "" : "/" + path);
      id = fragment.getParentId();
    }
    return path;
  }

  private void assertExpandedActions(String format, boolean aspect) throws Exception {
    setupMapDirectory();
    if (aspect) {
      addOptions("--aspects=//mapped:defs.bzl%mapped_aspect", "--output_groups=mapped");
    }
    buildTarget("//mapped:example");
    ActionGraphContainer graph =
        dumpSkyframe(format, "outputs('.*/(a|b)$', mnemonic('ExpandedCopy'))");
    // Each of the two children writes to both trees but must appear only once.
    assertThat(graph.getActionsCount()).isEqualTo(2);
    Map<Integer, Artifact> artifacts = new HashMap<>();
    for (Artifact artifact : graph.getArtifactsList()) {
      artifacts.put(artifact.getId(), artifact);
    }
    for (Action action : graph.getActionsList()) {
      assertThat(action.getMnemonic()).isEqualTo("ExpandedCopy");
      assertThat(action.getActionKey()).isNotEmpty();
      assertThat(action.getArgumentsCount()).isEqualTo(5);
      assertThat(action.getOutputIdsCount()).isEqualTo(2);
      assertThat(action.getOutputIdsList()).contains(action.getPrimaryOutputId());
      assertThat(action.getInputDepSetIdsList()).isNotEmpty();
      assertThat(action.getConfigurationId()).isNotEqualTo(0);
      assertThat(action.getAspectDescriptorIdsCount()).isEqualTo(aspect ? 1 : 0);
      for (int i = 0; i < 2; i++) {
        Artifact output = artifacts.get(action.getOutputIds(i));
        assertThat(output.getIsTreeArtifact()).isFalse();
        assertThat(artifactPath(graph, output)).isEqualTo(action.getArguments(i + 3));
      }
      assertThat(
              graph.getTargetsList().stream()
                  .filter(target -> target.getId() == action.getTargetId())
                  .map(target -> target.getLabel())
                  .collect(toImmutableList()))
          .containsExactly("//mapped:example");
    }
    assertThat(
            dumpSkyframe(format, "outputs('not_a_child', mnemonic('ExpandedCopy'))")
                .getActionsList())
        .isEmpty();
  }

  @Test
  public void skyframeState_expandedTemplateChildren_proto() throws Exception {
    assertExpandedActions("proto", false);
  }

  @Test
  public void skyframeState_expandedTemplateChildren_json() throws Exception {
    assertExpandedActions("jsonproto", false);
  }

  @Test
  public void skyframeState_expandedTemplateChildren_textproto() throws Exception {
    assertExpandedActions("textproto", false);
  }

  @Test
  public void skyframeState_expandedAspectChildren() throws Exception {
    assertExpandedActions("proto", true);
  }

  @Test
  public void skyframeState_unexecutedTemplateHasNoChildren() throws Exception {
    setupMapDirectory();
    addOptions("--nobuild");
    buildTarget("//mapped:example");
    assertThat(
            dumpSkyframe("proto", "outputs('.*/(a|b)$', mnemonic('ExpandedCopy'))")
                .getActionsList())
        .isEmpty();
  }

  @Test
  public void targetQuery_doesNotIncludeExpandedTemplateChildren() throws Exception {
    setupMapDirectory();
    buildTarget("//mapped:example");
    ByteArrayOutputStream stdout = captureReporterStdout();
    addOptions("--output=proto");
    runtimeWrapper.runAqueryExprCommand("//mapped:example");
    ActionGraphContainer graph =
        ActionGraphContainer.parseFrom(stdout.toByteArray(), ExtensionRegistry.getEmptyRegistry());
    assertThat(
            graph.getActionsList().stream()
                .filter(action -> !action.getActionKey().isEmpty())
                .map(Action::getMnemonic)
                .collect(toImmutableList()))
        .doesNotContain("ExpandedCopy");
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

  @Test
  public void testAquery_treeArtifactInputWithMapEachAndExpandDirectories_bypassesMapEach()
      throws Exception {
    write(
        "tree_test/defs.bzl",
        """
        def _get_path(file):
            if file.is_directory:
                fail("we should never see directory in map_each when expand_directories is true")
            return file.path

        def _impl(ctx):
            dir = ctx.actions.declare_directory("dir")
            ctx.actions.run_shell(outputs = [dir], command = "touch %s/file" % dir.path)

            file = ctx.actions.declare_file(ctx.label.name + ".out")
            args = ctx.actions.args()
            args.add_all([dir], map_each = _get_path)
            ctx.actions.run_shell(
                inputs = [dir],
                outputs = [file],
                command = 'echo "$@" > %s' % file.path,
                arguments = [args],
            )
            return DefaultInfo(files = depset([file]))

        r = rule(implementation = _impl)
        """);
    write(
        "tree_test/BUILD",
        """
        load(":defs.bzl", "r")

        r(name = "a")
        """);

    ByteArrayOutputStream stdout = captureReporterStdout();
    addOptions("--output=text", "--include_commandline=true");
    runtimeWrapper.runAqueryExprCommand("//tree_test:a");

    String outputText = stdout.toString(UTF_8);
    assertThat(outputText).contains("tree_test/dir");
    assertThat(outputText).doesNotContain("we should never see");
  }
}
