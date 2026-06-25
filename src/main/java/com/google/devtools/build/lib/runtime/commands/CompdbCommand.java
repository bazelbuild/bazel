// Copyright 2026 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.runtime.commands;

import static com.google.devtools.build.lib.runtime.Command.BuildPhase.ANALYZES;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.Action;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionGraph;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredAspect;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FileProvider;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.rules.cpp.CppCompileAction;
import com.google.devtools.build.lib.rules.cpp.CompilationDatabaseGenerator;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionPriority.PriorityCategory;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsClass;
import com.google.devtools.common.options.OptionsParsingException;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import java.io.IOException;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * Handles the 'compdb' command on the Bazel command line.
 *
 * <p>This command generates a compile_commands.json file for C/C++ targets, which can be used by
 * tools like clangd, clang-tidy, and other C/C++ development tools.
 *
 * <p>To use, run:
 *
 * <pre>
 * bazel compdb //target
 * </pre>
 *
 * <p>The compile_commands.json file will be written to the specified output path (default:
 * compile_commands.json in the workspace root).
 */
@Command(
    name = "compdb",
    buildPhase = ANALYZES,
    inheritsOptionsFrom = {BuildCommand.class},
    options = {CompdbCommand.CompdbOptions.class},
    usesConfigurationOptions = true,
    shortDescription = "Generates compile_commands.json for C/C++ targets.",
    completion = "label",
    help = "resource:compdb.txt",
    allowResidue = true)
public class CompdbCommand implements BlazeCommand {

  /** Options for the compdb command. */
  @OptionsClass
  public abstract static class CompdbOptions extends OptionsBase {
    @Option(
        name = "compdb_output",
        defaultValue = "compile_commands.json",
        documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
        effectTags = {OptionEffectTag.AFFECTS_OUTPUTS},
        help = "Output path for the compile_commands.json file. Use '-' for stdout.")
    public abstract String getCompdbOutput();
  }

  /**
   * Post-analysis processor that collects CppCompileActions from the analyzed action graph without
   * triggering actual compilation.
   */
  private static class CompdbProcessor implements BuildTool.AnalysisPostProcessor {

    private final List<CompilationDatabaseGenerator.Entry> entries = new ArrayList<>();
    private final Path workspaceRoot;

    CompdbProcessor(Path workspaceRoot) {
      this.workspaceRoot = workspaceRoot;
    }

    @Override
    public void process(
        BuildRequest request,
        CommandEnvironment env,
        BlazeRuntime runtime,
        AnalysisResult analysisResult)
        throws InterruptedException, ViewCreationFailedException {

      Set<Action> processedActions = new HashSet<>();
      ActionGraph actionGraph = analysisResult.getActionGraph();

      // Walk the action graph backwards from each requested target's outputs through link/archive
      // inputs to collect all transitive CppCompileActions (same scope as hedron's deps(target)).
      // Note: analysisResult.getArtifactsToBuild() is empty under --nobuild, so we cannot start
      // from that set.
      for (ConfiguredTarget configuredTarget : analysisResult.getTargetsToBuild()) {
        for (Artifact artifact :
            configuredTarget.getProvider(FileProvider.class).getFilesToBuild().toList()) {
          collectFromArtifact(artifact, actionGraph, processedActions);
        }
      }

      // Collect from aspects applied to targets. Aspects register their own actions directly.
      for (ConfiguredAspect aspect : analysisResult.getAspectsMap().values()) {
        collectFromAspect(aspect, processedActions);
      }
    }

    /**
     * Recursively walks the action graph from a target's outputs. For CppCompileActions, records the
     * compile entry. For other actions (e.g. link, archive), recurses through their inputs to find
     * underlying compile actions in transitive dependencies.
     */
    private void collectFromArtifact(
        Artifact artifact, ActionGraph actionGraph, Set<Action> processedActions)
        throws ViewCreationFailedException, InterruptedException {

      ActionAnalysisMetadata actionMetadata = actionGraph.getGeneratingAction(artifact);
      if (!(actionMetadata instanceof Action action)) {
        return;
      }

      if (!processedActions.add(action)) {
        return;
      }

      if (action instanceof CppCompileAction cppAction) {
        addEntry(cppAction);
        return;
      }

      for (Artifact input : action.getInputs().toList()) {
        collectFromArtifact(input, actionGraph, processedActions);
      }
    }

    private void collectFromAspect(
        ConfiguredAspect aspect, Set<Action> processedActions)
        throws ViewCreationFailedException, InterruptedException {

      for (ActionAnalysisMetadata action : aspect.getActions()) {
        if (action instanceof CppCompileAction cppAction) {
          if (processedActions.contains((Action) cppAction)) {
            continue;
          }
          processedActions.add((Action) cppAction);
          addEntry(cppAction);
        }
      }
    }

    private void addEntry(CppCompileAction cppAction)
        throws ViewCreationFailedException, InterruptedException {
      try {
        List<String> arguments = cppAction.getArguments();

        String output = null;
        if (!cppAction.getOutputs().isEmpty()) {
          output = cppAction.getOutputs().iterator().next().getExecPathString();
        }

        Artifact sourceFile = cppAction.getSourceFile();
        // Workspace root as directory; file paths are exec-root-relative (e.g. bin/foo.cc,
        // external/repo+/bar.cc) so they resolve via bazel-out/ and external/ symlinks — same as
        // hedron compile_commands.
        String filePath = sourceFile.getExecPathString();

        entries.add(
            new CompilationDatabaseGenerator.Entry(
                workspaceRoot.getPathString(),
                filePath,
                arguments,
                output));
      } catch (CommandLineExpansionException e) {
        throw new ViewCreationFailedException(
            "Failed to expand command line: " + e.getMessage(),
            FailureDetail.newBuilder()
                .setMessage("Failed to expand command line: " + e.getMessage())
                .build(),
            e);
      }
    }
  }

  @Override
  public void editOptions(OptionsParser optionsParser) {
    // Inject --nobuild to skip the execution phase. We only need the action graph from analysis
    // to extract compile commands, so actual compilation is not necessary.
    try {
      optionsParser.parse(
          PriorityCategory.COMPUTED_DEFAULT,
          "Option required by compdb",
          ImmutableList.of("--nobuild"));
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("Compdb's known options failed to parse", e);
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    BlazeRuntime runtime = env.getRuntime();
    CompdbOptions compdbOptions = options.getOptions(CompdbOptions.class);

    List<String> targets;
    try {
      targets = TargetPatternsHelper.readFrom(env, options);
    } catch (TargetPatternsHelper.TargetPatternsHelperException e) {
      env.getReporter().handle(Event.error(e.getMessage()));
      return BlazeCommandResult.failureDetail(e.getFailureDetail());
    }

    if (targets.isEmpty()) {
      env.getReporter()
          .handle(
              Event.warn(
                  "Usage: "
                      + runtime.getProductName()
                      + " compdb <options> <targets>."
                      + "\nInvoke `"
                      + runtime.getProductName()
                      + " help compdb` for full description of usage and options."
                      + "\nYour request is correct, but requested an empty set of targets."
                      + " Nothing will be built."));
    }

    BuildRequest request;
    try {
      request =
          BuildRequest.builder()
              .setCommandName(getClass().getAnnotation(Command.class).name())
              .setId(env.getCommandId())
              .setOptions(options)
              .setStartupOptions(runtime.getStartupOptionsProvider())
              .setOutErr(env.getReporter().getOutErr())
              .setTargets(targets)
              .setStartTimeMillis(env.getCommandStartTime())
              .build();
    } catch (IllegalArgumentException e) {
      env.getReporter().handle(Event.error("Failed to build request: " + e.getMessage()));
      FailureDetail failureDetail =
          FailureDetail.newBuilder()
              .setMessage("Failed to build request: " + e.getMessage())
              .build();
      return BlazeCommandResult.failureDetail(failureDetail);
    }

    // Use a custom processor that collects compile commands from the action graph
    // after analysis but before execution (which is skipped via --nobuild).
    Path workspaceRoot = env.getWorkspace();
    CompdbProcessor processor = new CompdbProcessor(workspaceRoot);
    BuildTool buildTool = new BuildTool(env, processor);
    BuildResult result = buildTool.processRequest(request, null, options);

    if (!result.getDetailedExitCode().isSuccess()) {
      return BlazeCommandResult.detailedExitCode(result.getDetailedExitCode());
    }

    try {
      writeCompilationDatabase(env, compdbOptions, processor.entries);
    } catch (IOException e) {
      env.getReporter()
          .handle(
              Event.error(
                  "Failed to generate compile_commands.json: " + e.getMessage()));
      FailureDetail failureDetail =
          FailureDetail.newBuilder()
              .setMessage("Failed to generate compile_commands.json: " + e.getMessage())
              .build();
      return BlazeCommandResult.failureDetail(failureDetail);
    }

    return BlazeCommandResult.success();
  }

  private void writeCompilationDatabase(
      CommandEnvironment env,
      CompdbOptions compdbOptions,
      List<CompilationDatabaseGenerator.Entry> entries)
      throws IOException {

    String outputPath = compdbOptions.getCompdbOutput();
    if (outputPath.equals("-")) {
      String json = CompilationDatabaseGenerator.toJson(entries);
      env.getReporter().getOutErr().printOut(json);
    } else {
      Path outputFile;
      if (outputPath.startsWith("/")) {
        outputFile = env.getExecRoot().getFileSystem().getPath(outputPath);
      } else {
        outputFile = env.getWorkspace().getRelative(outputPath);
      }
      byte[] jsonBytes = CompilationDatabaseGenerator.toJsonBytes(entries);
      outputFile.getParentDirectory().createDirectoryAndParents();
      try (OutputStream os = outputFile.getOutputStream()) {
        os.write(jsonBytes);
      }
      env.getReporter()
          .handle(
              Event.info(
                  "Generated compile_commands.json with " + entries.size() + " entries"));
    }
  }
}
