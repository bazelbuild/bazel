// Copyright 2014 The Bazel Authors. All rights reserved.
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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Strings;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSortedSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPathResolver;
import com.google.devtools.build.lib.actions.CommandLine;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.EnvironmentalExecException;
import com.google.devtools.build.lib.actions.ExecException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.FilesToRunProvider;
import com.google.devtools.build.lib.analysis.RunfilesSupport;
import com.google.devtools.build.lib.analysis.ShToolchain;
import com.google.devtools.build.lib.analysis.config.BuildConfiguration;
import com.google.devtools.build.lib.analysis.config.CoreOptions;
import com.google.devtools.build.lib.analysis.config.RunUnder;
import com.google.devtools.build.lib.analysis.test.TestConfiguration;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.analysis.test.TestRunnerAction;
import com.google.devtools.build.lib.analysis.test.TestStrategy;
import com.google.devtools.build.lib.analysis.test.TestTargetExecutionSettings;
import com.google.devtools.build.lib.buildtool.BuildRequest;
import com.google.devtools.build.lib.buildtool.BuildRequestOptions;
import com.google.devtools.build.lib.buildtool.BuildResult;
import com.google.devtools.build.lib.buildtool.BuildTool;
import com.google.devtools.build.lib.buildtool.OutputDirectoryLinksUtils;
import com.google.devtools.build.lib.buildtool.PathPrettyPrinter;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.exec.SymlinkTreeHelper;
import com.google.devtools.build.lib.exec.TestPolicy;
import com.google.devtools.build.lib.packages.InputFile;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.packages.NonconfigurableAttributeMapper;
import com.google.devtools.build.lib.packages.OutputFile;
import com.google.devtools.build.lib.packages.Rule;
import com.google.devtools.build.lib.packages.Target;
import com.google.devtools.build.lib.packages.TargetUtils;
import com.google.devtools.build.lib.packages.Type;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.runtime.BlazeCommand;
import com.google.devtools.build.lib.runtime.BlazeCommandResult;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.BlazeServerStartupOptions;
import com.google.devtools.build.lib.runtime.Command;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.CommandProtos.EnvironmentVariable;
import com.google.devtools.build.lib.server.CommandProtos.ExecRequest;
import com.google.devtools.build.lib.server.FailureDetails;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted;
import com.google.devtools.build.lib.server.FailureDetails.RunCommand.Code;
import com.google.devtools.build.lib.shell.ShellUtils;
import com.google.devtools.build.lib.util.CommandDescriptionForm;
import com.google.devtools.build.lib.util.CommandFailureUtils;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.FileType;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.util.OS;
import com.google.devtools.build.lib.util.OptionsUtils;
import com.google.devtools.build.lib.util.ShellEscaper;
import com.google.devtools.build.lib.util.io.OutErr;
import com.google.devtools.build.lib.vfs.FileSystemUtils;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionDocumentationCategory;
import com.google.devtools.common.options.OptionEffectTag;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import com.google.devtools.common.options.OptionsParsingResult;
import com.google.protobuf.ByteString;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import javax.annotation.Nullable;

/**
 * Builds and run a target with the given command line arguments.
 */
@Command(name = "run",
         builds = true,
         options = { RunCommand.RunOptions.class },
         inherits = { BuildCommand.class },
         shortDescription = "Runs the specified target.",
         help = "resource:run.txt",
         allowResidue = true,
         hasSensitiveResidue = true,
         completion = "label-bin")
public class RunCommand implements BlazeCommand  {
  /** Options for the "run" command. */
  public static class RunOptions extends OptionsBase {
    @Option(
      name = "script_path",
      defaultValue = "null",
      documentationCategory = OptionDocumentationCategory.OUTPUT_PARAMETERS,
      effectTags = {OptionEffectTag.AFFECTS_OUTPUTS, OptionEffectTag.EXECUTION},
      converter = OptionsUtils.PathFragmentConverter.class,
      help =
          "If set, write a shell script to the given file which invokes the target. "
              + "If this option is set, the target is not run from %{product}. "
              + "Use '%{product} run --script_path=foo //foo && ./foo' to invoke target '//foo' "
              + "This differs from '%{product} run //foo' in that the %{product} lock is released "
              + "and the executable is connected to the terminal's stdin."
    )
    public PathFragment scriptPath;
  }

  // Thrown when a method needs Bash but ShToolchain.getPath yields none.
  private static final class NoShellFoundException extends Exception {}

  @VisibleForTesting
  public static final String NO_TARGET_MESSAGE = "No targets found to run";

  public static final String MULTIPLE_TESTS_MESSAGE =
      "'run' only works with tests with one shard ('--test_sharding_strategy=disabled' is okay) "
      + "and without --runs_per_test";

  // The test policy to determine the environment variables from when running tests
  private final TestPolicy testPolicy;

  // Value of --run_under as of the most recent command invocation.
  private RunUnder currentRunUnder;

  private static final FileType RUNFILES_MANIFEST = FileType.of(".runfiles_manifest");

  public RunCommand(TestPolicy testPolicy) {
    this.testPolicy = testPolicy;
  }

  @VisibleForTesting  // productionVisibility = Visibility.PRIVATE
  protected BuildResult processRequest(final CommandEnvironment env, BuildRequest request) {
    List<String> targetPatternStrings = request.getTargets();
    return new BuildTool(env)
        .processRequest(
            request,
            (Collection<Target> targets, boolean keepGoing) ->
                RunCommand.this.validateTargets(
                    env.getReporter(), targetPatternStrings, targets, keepGoing));
  }

  @Override
  public void editOptions(OptionsParser optionsParser) { }

  /**
   * Compute the arguments the binary should be run with by concatenating the arguments in its
   * {@code args} attribute and the arguments on the Blaze command line.
   */
  // TODO(bazel-team): audit the use of null values by caller. It looks unsafe.
  @Nullable
  private List<String> computeArgs(
      CommandEnvironment env, ConfiguredTarget targetToRun, List<String> commandLineArgs) {
    List<String> args = Lists.newArrayList();

    FilesToRunProvider provider = targetToRun.getProvider(FilesToRunProvider.class);
    RunfilesSupport runfilesSupport = provider == null ? null : provider.getRunfilesSupport();
    if (runfilesSupport != null && runfilesSupport.getArgs() != null) {
      CommandLine targetArgs = runfilesSupport.getArgs();
      try {
        Iterables.addAll(args, targetArgs.arguments());
      } catch (InterruptedException ex) {
        // TODO(b/173521404): report a specific FailureDetail for "interrupted".
        env.getReporter().handle(Event.error("Interrupted while expanding target command line"));
        return null;
      } catch (CommandLineExpansionException e) {
        env.getReporter().handle(Event.error("Could not expand target command line: " + e));
        return null;
      }
    }
    args.addAll(commandLineArgs);
    return args;
  }

  private void constructCommandLine(
      List<String> cmdLine,
      List<String> prettyCmdLine,
      CommandEnvironment env,
      BuildConfiguration configuration,
      ConfiguredTarget targetToRun,
      ConfiguredTarget runUnderTarget,
      List<String> args)
      throws NoShellFoundException {
    BlazeRuntime runtime = env.getRuntime();
    String productName = runtime.getProductName();
    Artifact executable = targetToRun.getProvider(FilesToRunProvider.class).getExecutable();

    BuildRequestOptions requestOptions = env.getOptions().getOptions(BuildRequestOptions.class);

    PathFragment executablePath = executable.getPath().asFragment();
    PathPrettyPrinter prettyPrinter =
        OutputDirectoryLinksUtils.getPathPrettyPrinter(
            runtime.getRuleClassProvider().getSymlinkDefinitions(),
            requestOptions.getSymlinkPrefix(productName),
            productName,
            env.getWorkspace(),
            requestOptions.printWorkspaceInOutputPathsIfNeeded
                ? env.getWorkingDirectory()
                : env.getWorkspace(),
            requestOptions.experimentalNoProductNameOutSymlink);
    PathFragment prettyExecutablePath = prettyPrinter.getPrettyPath(executable.getPath());

    RunUnder runUnder = env.getOptions().getOptions(CoreOptions.class).runUnder;
    // Insert the command prefix specified by the "--run_under=<command-prefix>" option
    // at the start of the command line.
    if (runUnder != null) {
      String runUnderValue = runUnder.getValue();
      if (runUnderTarget != null) {
        // --run_under specifies a target. Get the corresponding executable.
        // This must be an absolute path, because the run_under target is only
        // in the runfiles of test targets.
        runUnderValue = runUnderTarget
            .getProvider(FilesToRunProvider.class).getExecutable().getPath().getPathString();
        // If the run_under command contains any options, make sure to add them
        // to the command line as well.
        List<String> opts = runUnder.getOptions();
        if (!opts.isEmpty()) {
          runUnderValue += " " + ShellEscaper.escapeJoinAll(opts);
        }
      }

      PathFragment shellExecutable = ShToolchain.getPath(configuration);
      if (shellExecutable.isEmpty()) {
        throw new NoShellFoundException();
      }

      cmdLine.add(shellExecutable.getPathString());
      cmdLine.add("-c");
      cmdLine.add(runUnderValue + " " + executablePath.getPathString() + " "
          + ShellEscaper.escapeJoinAll(args));
      prettyCmdLine.add(shellExecutable.getPathString());
      prettyCmdLine.add("-c");
      prettyCmdLine.add(runUnderValue + " " + prettyExecutablePath.getPathString() + " "
          + ShellEscaper.escapeJoinAll(args));
    } else {
      cmdLine.add(executablePath.getPathString());
      cmdLine.addAll(args);
      prettyCmdLine.add(prettyExecutablePath.getPathString());
      prettyCmdLine.addAll(args);
    }
  }

  @Override
  public BlazeCommandResult exec(CommandEnvironment env, OptionsParsingResult options) {
    RunOptions runOptions = options.getOptions(RunOptions.class);
    // This list should look like: ["//executable:target", "arg1", "arg2"]
    List<String> targetAndArgs = options.getResidue();

    // The user must at the least specify an executable target.
    if (targetAndArgs.isEmpty()) {
      return reportAndCreateFailureResult(
          env, "Must specify a target to run", Code.NO_TARGET_SPECIFIED);
    }
    String targetString = targetAndArgs.get(0);
    List<String> commandLineArgs = targetAndArgs.subList(1, targetAndArgs.size());
    RunUnder runUnder = options.getOptions(CoreOptions.class).runUnder;

    OutErr outErr = env.getReporter().getOutErr();
    List<String> targets = (runUnder != null) && (runUnder.getLabel() != null)
        ? ImmutableList.of(targetString, runUnder.getLabel().toString())
        : ImmutableList.of(targetString);
    BuildRequest request = BuildRequest.create(
        this.getClass().getAnnotation(Command.class).name(), options,
        env.getRuntime().getStartupOptionsProvider(), targets, outErr,
        env.getCommandId(), env.getCommandStartTime());

    currentRunUnder = runUnder;
    BuildResult result;
    try {
      result = processRequest(env, request);
    } finally {
      currentRunUnder = null;
    }

    if (!result.getSuccess()) {
      env.getReporter().handle(Event.error("Build failed. Not running target"));
      return BlazeCommandResult.detailedExitCode(result.getDetailedExitCode());
    }

    // Make sure that we have exactly 1 built target (excluding --run_under),
    // and that it is executable.
    // These checks should only fail if keepGoing is true, because we already did
    // validation before the build began.  See {@link #validateTargets()}.
    Collection<ConfiguredTarget> targetsBuilt = result.getSuccessfulTargets();
    ConfiguredTarget targetToRun = null;
    ConfiguredTarget runUnderTarget = null;

    if (targetsBuilt != null) {
      int maxTargets = runUnder != null && runUnder.getLabel() != null ? 2 : 1;
      if (targetsBuilt.size() > maxTargets) {
        return reportAndCreateFailureResult(
            env,
            makeErrorMessageForNotHavingASingleTarget(
                targetString, Iterables.transform(targetsBuilt, ct -> ct.getLabel().toString())),
            Code.TOO_MANY_TARGETS_SPECIFIED);
      }
      for (ConfiguredTarget target : targetsBuilt) {
        BlazeCommandResult targetValidation = fullyValidateTarget(env, target);
        if (!targetValidation.isSuccess()) {
          return targetValidation;
        }
        if (runUnder != null && target.getLabel().equals(runUnder.getLabel())) {
          if (runUnderTarget != null) {
            return reportAndCreateFailureResult(
                env,
                "Can't identify the run_under target from multiple options?",
                Code.RUN_UNDER_TARGET_NOT_BUILT);
          }
          runUnderTarget = target;
        } else if (targetToRun == null) {
          targetToRun = target;
        } else {
          return reportAndCreateFailureResult(
              env,
              makeErrorMessageForNotHavingASingleTarget(
                  targetString, Iterables.transform(targetsBuilt, ct -> ct.getLabel().toString())),
              Code.TOO_MANY_TARGETS_SPECIFIED);
        }
      }
    }

    // Handle target & run_under referring to the same target.
    if (targetToRun == null && runUnderTarget != null) {
      targetToRun = runUnderTarget;
    }

    if (targetToRun == null) {
      return reportAndCreateFailureResult(env, NO_TARGET_MESSAGE, Code.NO_TARGET_SPECIFIED);
    }

    BuildConfiguration configuration =
        env.getSkyframeExecutor()
            .getConfiguration(env.getReporter(), targetToRun.getConfigurationKey());
    if (configuration == null) {
      // The target may be an input file, which doesn't have a configuration. In that case, we
      // choose any target configuration.
      configuration = result.getBuildConfigurationCollection().getTargetConfigurations().get(0);
    }

    if (!configuration.buildRunfilesManifests()) {
      return reportAndCreateFailureResult(
          env,
          "--nobuild_runfile_manifests is incompatible with the \"run\" command",
          Code.RUN_PREREQ_UNMET);
    }

    Path runfilesDir;
    FilesToRunProvider provider = targetToRun.getProvider(FilesToRunProvider.class);
    RunfilesSupport runfilesSupport = provider == null ? null : provider.getRunfilesSupport();

    if (runfilesSupport == null) {
      runfilesDir = env.getWorkingDirectory();
    } else {
      try {
        runfilesDir = ensureRunfilesBuilt(env, runfilesSupport,
            env.getSkyframeExecutor().getConfiguration(env.getReporter(),
                targetToRun.getConfigurationKey()));
      } catch (RunfilesException e) {
        env.getReporter().handle(Event.error(e.getMessage()));
        return BlazeCommandResult.failureDetail(e.createFailureDetail());
      } catch (InterruptedException e) {
        env.getReporter().handle(Event.error("Interrupted"));
        return BlazeCommandResult.failureDetail(
            FailureDetail.newBuilder()
                .setInterrupted(Interrupted.newBuilder().setCode(Interrupted.Code.INTERRUPTED))
                .build());
      }
    }

    Map<String, String> runEnvironment = new TreeMap<>();
    List<String> cmdLine = new ArrayList<>();
    List<String> prettyCmdLine = new ArrayList<>();
    Path workingDir;

    runEnvironment.put("BUILD_WORKSPACE_DIRECTORY", env.getWorkspace().getPathString());
    runEnvironment.put("BUILD_WORKING_DIRECTORY", env.getWorkingDirectory().getPathString());

    if (targetToRun.getProvider(TestProvider.class) != null) {
      // This is a test. Provide it with a reasonable approximation of the actual test environment
      ImmutableList<Artifact.DerivedArtifact> statusArtifacts =
          TestProvider.getTestStatusArtifacts(targetToRun);
      if (statusArtifacts.size() != 1) {
        return reportAndCreateFailureResult(
            env, MULTIPLE_TESTS_MESSAGE, Code.TOO_MANY_TEST_SHARDS_OR_RUNS);
      }

      TestRunnerAction testAction = (TestRunnerAction) env.getSkyframeExecutor()
          .getActionGraph(env.getReporter()).getGeneratingAction(
              Iterables.getOnlyElement(statusArtifacts));
      TestTargetExecutionSettings settings = testAction.getExecutionSettings();
      // ensureRunfilesBuilt does build the runfiles, but an extra consistency check won't hurt.
      Preconditions.checkState(
          settings.getRunfilesSymlinksCreated()
              == options.getOptions(CoreOptions.class).buildRunfiles);

      ExecutionOptions executionOptions = options.getOptions(ExecutionOptions.class);
      Path tmpDirRoot = TestStrategy.getTmpRoot(
          env.getWorkspace(), env.getExecRoot(), executionOptions);
      PathFragment maybeRelativeTmpDir =
          tmpDirRoot.startsWith(env.getExecRoot())
              ? tmpDirRoot.relativeTo(env.getExecRoot())
              : tmpDirRoot.asFragment();
      Duration timeout =
          configuration
              .getFragment(TestConfiguration.class)
              .getTestTimeout()
              .get(testAction.getTestProperties().getTimeout());
      runEnvironment.putAll(
          testPolicy.computeTestEnvironment(
              testAction,
              env.getClientEnv(),
              timeout,
              settings.getRunfilesDir().relativeTo(env.getExecRoot()),
              maybeRelativeTmpDir.getRelative(TestStrategy.getTmpDirName(testAction))));
      workingDir = env.getExecRoot();

      try {
        testAction.prepare(env.getExecRoot(), ArtifactPathResolver.IDENTITY, /*bulkDeleter=*/ null);
      } catch (IOException e) {
        return reportAndCreateFailureResult(
            env,
            "Error while setting up test: " + e.getMessage(),
            Code.TEST_ENVIRONMENT_SETUP_FAILURE);
      } catch (InterruptedException e) {
        return reportAndCreateFailureResult(
            env,
            "Error while setting up test: " + e.getMessage(),
            Code.TEST_ENVIRONMENT_SETUP_INTERRUPTED);
      }

      try {
        cmdLine.addAll(TestStrategy.getArgs(testAction));
        cmdLine.addAll(commandLineArgs);
        prettyCmdLine.addAll(cmdLine);
      } catch (ExecException e) {
        return reportAndCreateFailureResult(
            env, Strings.nullToEmpty(e.getMessage()), Code.COMMAND_LINE_EXPANSION_FAILURE);
      } catch (InterruptedException e) {
        String message = "run: argument expansion interrupted";
        env.getReporter().handle(Event.error(message));
        return BlazeCommandResult.detailedExitCode(
            InterruptedFailureDetails.detailedExitCode(message));
      }
    } else {
      workingDir = runfilesDir;
      List<String> args = computeArgs(env, targetToRun, commandLineArgs);
      try {
        constructCommandLine(
            cmdLine, prettyCmdLine, env, configuration, targetToRun, runUnderTarget, args);
      } catch (NoShellFoundException e) {
        return reportAndCreateFailureResult(
            env,
            "the \"run\" command needs a shell with \"--run_under\"; use the"
                + " --shell_executable=<path> flag to specify its path, e.g."
                + " --shell_executable=/bin/bash",
            Code.NO_SHELL_SPECIFIED);
      }
    }

    if (runOptions.scriptPath != null) {
      String unisolatedCommand = CommandFailureUtils.describeCommand(
          CommandDescriptionForm.COMPLETE_UNISOLATED,
          /* prettyPrintArgs= */ false,
          cmdLine,
          runEnvironment,
          workingDir.getPathString());

      PathFragment shExecutable = ShToolchain.getPath(configuration);
      if (shExecutable.isEmpty()) {
        return reportAndCreateFailureResult(
            env,
            "the \"run\" command needs a shell with \"--script_path\"; use the"
                + " --shell_executable=<path> flag to specify its path, e.g."
                + " --shell_executable=/bin/bash",
            Code.NO_SHELL_SPECIFIED);
      }

      try {
        writeScript(env, shExecutable, runOptions.scriptPath, unisolatedCommand);
        return BlazeCommandResult.success();
      } catch (IOException e) {
        String message = "Error writing run script: " + e.getMessage();
        return reportAndCreateFailureResult(env, message, Code.SCRIPT_WRITE_FAILURE);
      }
    }

    // We need to do update runEnvironment so that the environment of --batch is not contaminated
    // with that required for the server. Note that some differences between the environment of
    // the process being run and the environment of the client are still possible if the environment
    // variables added for the server were not in the original client environment.
    //
    // This is done after writing the script for --script_path so that that is not contaminated
    // with the original client environment (CommandFailureUtils.describeCommand() puts
    // runEnvironment into the written script)
    boolean batchMode = env.getRuntime().getStartupOptionsProvider()
        .getOptions(BlazeServerStartupOptions.class).batch;
    if (batchMode) {
      runEnvironment.putAll(env.getClientEnv());
    }

    env.getReporter().handle(Event.info(
        null, "Running command line: " + ShellEscaper.escapeJoinAll(prettyCmdLine)));

    ExecRequest.Builder execDescription = ExecRequest.newBuilder()
        .setWorkingDirectory(
            ByteString.copyFrom(workingDir.getPathString(), StandardCharsets.ISO_8859_1));

    if (OS.getCurrent() == OS.WINDOWS) {
      boolean isBinary = true;
      for (String arg : cmdLine) {
        if (!isBinary) {
          // All but the first element in `cmdLine` have to be escaped. The first element is the
          // binary, which must not be escaped.
          arg = ShellUtils.windowsEscapeArg(arg);
        }
        execDescription.addArgv(ByteString.copyFrom(arg, StandardCharsets.ISO_8859_1));
        isBinary = false;
      }
    } else {
      PathFragment shExecutable = ShToolchain.getPath(configuration);
      if (shExecutable.isEmpty()) {
        return reportAndCreateFailureResult(
            env,
            "the \"run\" command needs a shell with; use the --shell_executable=<path> "
                + "flag to specify the shell's path, e.g. --shell_executable=/bin/bash",
            Code.NO_SHELL_SPECIFIED);
      }

      String shellEscaped = ShellEscaper.escapeJoinAll(cmdLine);
      if (OS.getCurrent() == OS.WINDOWS) {
        // On Windows, we run Bash as a subprocess of the client (via CreateProcessW).
        // Bash uses its own (Bash-style) flag parsing logic, not the default logic for which
        // ShellUtils.windowsEscapeArg escapes, so we escape the flags once again Bash-style.
        shellEscaped = "\"" + shellEscaped.replace("\\", "\\\\").replace("\"", "\\\"") + "\"";
      }

      ImmutableList<String> shellCmdLine =
          ImmutableList.<String>of(shExecutable.getPathString(), "-c", shellEscaped);

      for (String arg : shellCmdLine) {
        execDescription.addArgv(ByteString.copyFrom(arg, StandardCharsets.ISO_8859_1));
      }
    }

    for (Map.Entry<String, String> variable : runEnvironment.entrySet()) {
      execDescription.addEnvironmentVariable(EnvironmentVariable.newBuilder()
          .setName(ByteString.copyFrom(variable.getKey(), StandardCharsets.ISO_8859_1))
          .setValue(ByteString.copyFrom(variable.getValue(), StandardCharsets.ISO_8859_1))
          .build());
    }

    return BlazeCommandResult.execute(execDescription.build());
  }

  private static BlazeCommandResult reportAndCreateFailureResult(
      CommandEnvironment env, String message, Code detailedCode) {
    env.getReporter().handle(Event.error(message));
    return BlazeCommandResult.failureDetail(createFailureDetail(message, detailedCode));
  }

  /**
   * Ensures that runfiles are built for the specified target. If they already are, does nothing,
   * otherwise builds them.
   */
  private static Path ensureRunfilesBuilt(
      CommandEnvironment env, RunfilesSupport runfilesSupport, BuildConfiguration configuration)
      throws RunfilesException, InterruptedException {
    Artifact manifest = Preconditions.checkNotNull(runfilesSupport.getRunfilesManifest());
    PathFragment runfilesDir = runfilesSupport.getRunfilesDirectoryExecPath();
    Path workingDir = env.getExecRoot().getRelative(runfilesDir);
    // On Windows, runfiles tree is disabled.
    // Workspace name directory doesn't exist, so don't add it.
    if (configuration.runfilesEnabled()) {
      workingDir = workingDir.getRelative(runfilesSupport.getRunfiles().getSuffix());
    }

    // Always create runfiles directory and the workspace-named directory underneath, even if we
    // run with --enable_runfiles=no (which is the default on Windows as of 2020-01-24).
    // If the binary we run is in fact a test, it will expect to be able to chdir into the runfiles
    // directory. See https://github.com/bazelbuild/bazel/issues/10621
    try {
      runfilesSupport
          .getRunfilesDirectory()
          .getRelative(runfilesSupport.getWorkspaceName())
          .createDirectoryAndParents();
    } catch (IOException e) {
      throw new RunfilesException(
          "Failed to create runfiles directories: " + e.getMessage(),
          Code.RUNFILES_DIRECTORIES_CREATION_FAILURE,
          e);
    }

    // When runfiles are not generated, getManifest() returns the
    // .runfiles_manifest file, otherwise it returns the MANIFEST file. This is
    // a handy way to check whether runfiles were built or not.
    if (!RUNFILES_MANIFEST.matches(manifest.getFilename())) {
      return workingDir;
    }

    SymlinkTreeHelper helper = new SymlinkTreeHelper(
        manifest.getPath(),
        runfilesSupport.getRunfilesDirectory(),
        false);
    try {
      helper.createSymlinksUsingCommand(
          env.getExecRoot(),
          env.getBlazeWorkspace().getBinTools(),
          /* shellEnvironment= */ ImmutableMap.of(),
          /* outErr= */ null);
    } catch (EnvironmentalExecException e) {
      throw new RunfilesException(
          "Failed to create runfiles symlinks: " + e.getMessage(),
          Code.RUNFILES_SYMLINKS_CREATION_FAILURE,
          e);
    }
    return workingDir;
  }

  private static void writeScript(
      CommandEnvironment env, PathFragment shellExecutable, PathFragment scriptPathFrag, String cmd)
      throws IOException {
    Path scriptPath = env.getWorkingDirectory().getRelative(scriptPathFrag);
    if (OS.getCurrent() == OS.WINDOWS) {
      FileSystemUtils.writeContent(
          scriptPath, StandardCharsets.ISO_8859_1, "@echo off\n" + cmd + " %*");
      scriptPath.setExecutable(true);
    } else {
      FileSystemUtils.writeContent(
          scriptPath,
          StandardCharsets.ISO_8859_1,
          "#!" + shellExecutable.getPathString() + "\n" + cmd + " \"$@\"");
      scriptPath.setExecutable(true);
    }
  }

  // Make sure we are building exactly 1 binary target.
  // If keepGoing, we'll build all the targets even if they are non-binary.
  private void validateTargets(
      Reporter reporter,
      List<String> targetPatternStrings,
      Collection<Target> targets,
      boolean keepGoing)
      throws LoadingFailedException {
    Target targetToRun = null;
    Target runUnderTarget = null;

    boolean singleTargetWarningWasOutput = false;
    int maxTargets = currentRunUnder != null && currentRunUnder.getLabel() != null ? 2 : 1;
    if (targets.size() > maxTargets) {
      warningOrException(
          reporter,
          makeErrorMessageForNotHavingASingleTarget(
              targetPatternStrings.get(0),
              Iterables.transform(targets, t -> t.getLabel().toString())),
          keepGoing,
          Code.TOO_MANY_TARGETS_SPECIFIED);
      singleTargetWarningWasOutput = true;
    }
    for (Target target : targets) {
      if (!isExecutable(target)) {
        warningOrException(
            reporter, notExecutableError(target), keepGoing, Code.TARGET_NOT_EXECUTABLE);
      }

      if (currentRunUnder != null && target.getLabel().equals(currentRunUnder.getLabel())) {
        // It's impossible to have two targets with the same label.
        Preconditions.checkState(runUnderTarget == null);
        runUnderTarget = target;
      } else if (targetToRun == null) {
        targetToRun = target;
      } else {
        if (!singleTargetWarningWasOutput) {
          warningOrException(
              reporter,
              makeErrorMessageForNotHavingASingleTarget(
                  targetPatternStrings.get(0),
                  Iterables.transform(targets, t -> t.getLabel().toString())),
              keepGoing,
              Code.TOO_MANY_TARGETS_SPECIFIED);
        }
        return;
      }
    }
    // Handle target & run_under referring to the same target.
    if ((targetToRun == null) && (runUnderTarget != null)) {
      targetToRun = runUnderTarget;
    }
    if (targetToRun == null) {
      warningOrException(reporter, NO_TARGET_MESSAGE, keepGoing, Code.NO_TARGET_SPECIFIED);
    }
  }

  /**
   * If keepGoing, print a warning and return the given collection. Otherwise, throw
   * InvalidTargetException.
   */
  private void warningOrException(
      Reporter reporter, String message, boolean keepGoing, Code detailedCode)
      throws LoadingFailedException {
    if (keepGoing) {
      reporter.handle(Event.warn(message + ". Will continue anyway"));
    } else {
      throw new LoadingFailedException(
          message, DetailedExitCode.of(createFailureDetail(message, detailedCode)));
    }
  }

  private static String notExecutableError(Target target) {
    return "Cannot run target " + target.getLabel() + ": Not executable";
  }

  /**
   * Performs all available validation checks on an individual target.
   *
   * @param configuredTarget ConfiguredTarget to validate
   * @return BlazeCommandResult.exitCode(ExitCode.SUCCESS) if all checks succeeded, otherwise a
   *     result describing the failure.
   * @throws IllegalStateException if unable to find a target from the package manager.
   */
  private static BlazeCommandResult fullyValidateTarget(
      CommandEnvironment env, ConfiguredTarget configuredTarget) {

    Target target;
    try {
      target = env.getPackageManager().getTarget(env.getReporter(), configuredTarget.getLabel());
    } catch (InterruptedException e) {
      String message = "run command interrupted";
      env.getReporter().handle(Event.error(message));
      return BlazeCommandResult.detailedExitCode(
          InterruptedFailureDetails.detailedExitCode(message));
    } catch (NoSuchTargetException | NoSuchPackageException e) {
      env.getReporter().handle(Event.error("Failed to find a target to validate. " + e));
      throw new IllegalStateException("Failed to find a target to validate", e);
    }

    if (!isExecutable(target)) {
      return reportAndCreateFailureResult(
          env, notExecutableError(target), Code.TARGET_NOT_EXECUTABLE);
    }

    Artifact executable =
        Preconditions.checkNotNull(
                configuredTarget.getProvider(FilesToRunProvider.class), configuredTarget)
            .getExecutable();
    if (executable == null) {
      return reportAndCreateFailureResult(
          env, notExecutableError(target), Code.TARGET_NOT_EXECUTABLE);
    }

    Path executablePath = executable.getPath();
    try {
      if (!executablePath.exists() || !executablePath.isExecutable()) {
        return reportAndCreateFailureResult(
            env,
            "Non-existent or non-executable " + executablePath,
            Code.TARGET_BUILT_BUT_PATH_NOT_EXECUTABLE);
      }
    } catch (IOException e) {
      return reportAndCreateFailureResult(
          env,
          "Error checking " + executablePath.getPathString() + ": " + e.getMessage(),
          Code.TARGET_BUILT_BUT_PATH_VALIDATION_FAILED);
    }

    return BlazeCommandResult.success();
  }

  /**
   * Return true iff it is possible that {@code target} is a rule that has an executable file. This
   * *_test rules, *_binary rules, aliases, generated outputs, and inputs.
   *
   * <p>Determining definitively whether a rule produces an executable can only be done after
   * analysis. This is only an early check to quickly catch most mistakes.
   */
  private static boolean isExecutable(Target target) {
    return isPlainFile(target)
        || isExecutableNonTestRule(target)
        || TargetUtils.isTestRule(target)
        || TargetUtils.isAlias(target);
  }

  /**
   * Return true iff {@code target} is a rule that generates an executable file and is user-executed
   * code.
   */
  private static boolean isExecutableNonTestRule(Target target) {
    if (!(target instanceof Rule)) {
      return false;
    }
    Rule rule = ((Rule) target);
    if (rule.getRuleClassObject().hasAttr("$is_executable", Type.BOOLEAN)) {
      return NonconfigurableAttributeMapper.of(rule).get("$is_executable", Type.BOOLEAN);
    }
    return false;
  }

  private static boolean isPlainFile(Target target) {
    return (target instanceof OutputFile) || (target instanceof InputFile);
  }

  private String makeErrorMessageForNotHavingASingleTarget(
      String targetPatternString, Iterable<String> expandedTargetNames) {
    final int maxNumExpandedTargetsToIncludeInErrorMessage = 5;
    boolean truncateTargetNameList = Iterables.size(expandedTargetNames) > 5;
    Iterable<String> targetNamesToIncludeInErrorMessage =
        truncateTargetNameList
            ? Iterables.limit(expandedTargetNames, maxNumExpandedTargetsToIncludeInErrorMessage)
            : expandedTargetNames;
    return String.format(
        "Only a single target can be run. Your target pattern %s expanded to the targets %s%s",
        targetPatternString,
        Joiner.on(", ").join(ImmutableSortedSet.copyOf(targetNamesToIncludeInErrorMessage)),
        truncateTargetNameList ? "[TRUNCATED]" : "");
  }

  private static FailureDetail createFailureDetail(String message, Code detailedCode) {
    return FailureDetail.newBuilder()
        .setMessage(message)
        .setRunCommand(FailureDetails.RunCommand.newBuilder().setCode(detailedCode))
        .build();
  }

  private static class RunfilesException extends Exception {
    private final FailureDetails.RunCommand.Code detailedCode;

    private RunfilesException(String message, Code detailedCode, Exception cause) {
      super("Error creating runfiles: " + message, cause);
      this.detailedCode = detailedCode;
    }

    private FailureDetail createFailureDetail() {
      return RunCommand.createFailureDetail(getMessage(), detailedCode);
    }
  }
}
