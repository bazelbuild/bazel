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
package com.google.devtools.build.lib.buildtool;

import static com.google.devtools.build.lib.platform.SuspendCounter.suspendCount;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.BuildInfoEvent;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.WorkspaceStatusAction.DummyEnvironment;
import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.analysis.config.InvalidConfigurationException;
import com.google.devtools.build.lib.buildeventstream.BuildEvent.LocalFile.LocalFileType;
import com.google.devtools.build.lib.buildeventstream.BuildEventArtifactUploader.UploadContext;
import com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil;
import com.google.devtools.build.lib.buildeventstream.BuildEventProtocolOptions;
import com.google.devtools.build.lib.buildtool.buildevent.BuildCompleteEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildInterruptedEvent;
import com.google.devtools.build.lib.buildtool.buildevent.BuildStartingEvent;
import com.google.devtools.build.lib.buildtool.buildevent.NoExecutionEvent;
import com.google.devtools.build.lib.buildtool.buildevent.StartingAqueryDumpAfterBuildEvent;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.OutputFilter;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.aquery.ActionGraphProtoV2OutputFormatterCallback;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.server.FailureDetails.Interrupted.Code;
import com.google.devtools.build.lib.skyframe.DetailedTargetParsingException;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.OptionsProvider;
import com.google.devtools.common.options.RegexPatternOption;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import javax.annotation.Nullable;

/**
 * Provides the bulk of the implementation of the 'blaze build' command.
 *
 * <p>The various concrete build command classes handle the command options and request setup, then
 * delegate the handling of the request (the building of targets) to this class.
 *
 * <p>The main entry point is {@link #buildTargets}.
 *
 * <p>Most of analysis is handled in {@link com.google.devtools.build.lib.analysis.BuildView}, and
 * execution in {@link ExecutionTool}.
 */
public class BuildTool {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  protected final CommandEnvironment env;
  protected final BlazeRuntime runtime;

  /**
   * Constructs a BuildTool.
   *
   * @param env a reference to the command environment of the currently executing command
   */
  public BuildTool(CommandEnvironment env) {
    this.env = env;
    this.runtime = env.getRuntime();
  }

  /**
   * The crux of the build system: builds the targets specified in the request.
   *
   * <p>Performs loading, analysis and execution for the specified set of targets, honoring the
   * configuration options in the BuildRequest. Returns normally iff successful, throws an exception
   * otherwise.
   *
   * <p>Callers must ensure that {@link #stopRequest} is called after this method, even if it
   * throws.
   *
   * <p>The caller is responsible for setting up and syncing the package cache.
   *
   * <p>During this function's execution, the actualTargets and successfulTargets fields of the
   * request object are set.
   *
   * @param request the build request that this build tool is servicing, which specifies various
   *     options; during this method's execution, the actualTargets and successfulTargets fields of
   *     the request object are populated
   * @param result the build result that is the mutable result of this build
   * @param validator target validator
   */
  public void buildTargets(BuildRequest request, BuildResult result, TargetValidator validator)
      throws BuildFailedException, InterruptedException, ViewCreationFailedException,
          TargetParsingException, LoadingFailedException, AbruptExitException,
          InvalidConfigurationException, TestExecException, ExitException,
          PostExecutionActionGraphDumpException {
    try (SilentCloseable c = Profiler.instance().profile("validateOptions")) {
      validateOptions(request);
    }
    BuildOptions buildOptions;
    try (SilentCloseable c = Profiler.instance().profile("createBuildOptions")) {
      buildOptions = runtime.createBuildOptions(request);
    }

    ExecutionTool executionTool = null;
    boolean catastrophe = false;
    try {
      try (SilentCloseable c = Profiler.instance().profile("BuildStartingEvent")) {
        env.getEventBus().post(new BuildStartingEvent(env, request));
      }
      logger.atInfo().log("Build identifier: %s", request.getId());

      // Error out early if multi_cpus is set, but we're not in build or test command.
      if (!request.getMultiCpus().isEmpty()) {
        getReporter().handle(Event.warn(
            "The --experimental_multi_cpu option is _very_ experimental and only intended for "
            + "internal testing at this time. If you do not work on the build tool, then you "
            + "should stop now!"));
        if (!"build".equals(request.getCommandName()) && !"test".equals(request.getCommandName())) {
          throw new InvalidConfigurationException(
              "The experimental setting to select multiple CPUs is only supported for 'build' and "
              + "'test' right now!");
        }
      }

      // Exit if there are any pending exceptions from modules.
      env.throwPendingException();

      initializeOutputFilter(request);

      AnalysisResult analysisResult =
          AnalysisPhaseRunner.execute(env, request, buildOptions, validator);

      // We cannot move the executionTool down to the execution phase part since it does set up the
      // symlinks for tools.
      // TODO(twerth): Extract embedded tool setup from execution tool and move object creation to
      // execution phase.
      executionTool = new ExecutionTool(env, request);
      if (request.getBuildOptions().performAnalysisPhase) {

        if (!analysisResult.getExclusiveTests().isEmpty()
            && executionTool.getTestActionContext().forceParallelTestExecution()) {
          String testStrategy = request.getOptions(ExecutionOptions.class).testStrategy;
          for (ConfiguredTarget test : analysisResult.getExclusiveTests()) {
            getReporter()
                .handle(
                    Event.warn(
                        test.getLabel()
                            + " is tagged exclusive, but --test_strategy="
                            + testStrategy
                            + " forces parallel test execution."));
          }
          analysisResult = analysisResult.withExclusiveTestsAsParallelTests();
        }

        result.setBuildConfigurationCollection(analysisResult.getConfigurationCollection());
        result.setActualTargets(analysisResult.getTargetsToBuild());
        result.setTestTargets(analysisResult.getTargetsToTest());

        try (SilentCloseable c = Profiler.instance().profile("postProcessAnalysisResult")) {
          postProcessAnalysisResult(request, analysisResult);
        }

        // Execution phase.
        if (needsExecutionPhase(request.getBuildOptions())) {
          try (SilentCloseable closeable = Profiler.instance().profile("ExecutionTool.init")) {
            executionTool.init();
          }
          executionTool.executeBuild(
              request.getId(),
              analysisResult,
              result,
              analysisResult.getPackageRoots(),
              request.getTopLevelArtifactContext());
        } else {
          env.getReporter().post(new NoExecutionEvent());
        }
        FailureDetail delayedFailureDetail = analysisResult.getFailureDetail();
        if (delayedFailureDetail != null) {
          throw new BuildFailedException(
              delayedFailureDetail.getMessage(), DetailedExitCode.of(delayedFailureDetail));
        }

        // Only consider builds with SequencedSkyframeExecutor.
        if (env.getSkyframeExecutor() instanceof SequencedSkyframeExecutor
            && request.getBuildOptions().aqueryDumpAfterBuildFormat != null) {
          try (SilentCloseable c = Profiler.instance().profile("postExecutionDumpSkyframe")) {
            dumpSkyframeStateAfterBuild(
                request.getOptions(BuildEventProtocolOptions.class),
                request.getBuildOptions().aqueryDumpAfterBuildFormat,
                request.getBuildOptions().aqueryDumpAfterBuildOutputFile);
          } catch (CommandLineExpansionException | IOException e) {
            throw new PostExecutionActionGraphDumpException(e);
          }
        }
      }
      Profiler.instance().markPhase(ProfilePhase.FINISH);
    } catch (Error | RuntimeException e) {
      request
          .getOutErr()
          .printErrLn(
              "Internal error thrown during build. Printing stack trace: "
                  + Throwables.getStackTraceAsString(e));
      catastrophe = true;
      throw e;
    } finally {
      if (executionTool != null) {
        executionTool.shutdown();
      }
      if (!catastrophe) {
        // Delete dirty nodes to ensure that they do not accumulate indefinitely.
        long versionWindow = request.getViewOptions().versionWindowForDirtyNodeGc;
        if (versionWindow != -1) {
          env.getSkyframeExecutor().deleteOldNodes(versionWindow);
        }
        // The workspace status actions will not run with certain flags, or if an error
        // occurs early in the build. Tell a lie so that the event is not missing.
        // If multiple build_info events are sent, only the first is kept, so this does not harm
        // successful runs (which use the workspace status action).
        env.getEventBus()
            .post(
                new BuildInfoEvent(
                    env.getBlazeWorkspace()
                        .getWorkspaceStatusActionFactory()
                        .createDummyWorkspaceStatus(
                            new DummyEnvironment() {
                              @Override
                              public Path getWorkspace() {
                                return env.getWorkspace();
                              }

                              @Override
                              public String getBuildRequestId() {
                                return env.getBuildRequestId();
                              }

                              @Override
                              public OptionsProvider getOptions() {
                                return env.getOptions();
                              }
                            })));
      }
    }
  }

  /**
   * This class is meant to be overridden by classes that want to perform the Analysis phase and
   * then process the results in some interesting way. See {@link CqueryBuildTool} as an example.
   */
  protected void postProcessAnalysisResult(BuildRequest request, AnalysisResult analysisResult)
      throws InterruptedException, ViewCreationFailedException, ExitException {}

  /**
   * Produces an aquery dump of the state of Skyframe.
   *
   * <p>There are 2 possible output channels: a local file or a remote FS.
   */
  private void dumpSkyframeStateAfterBuild(
      @Nullable BuildEventProtocolOptions besOptions,
      String format,
      @Nullable PathFragment outputFilePathFragment)
      throws CommandLineExpansionException, IOException {
    Preconditions.checkState(env.getSkyframeExecutor() instanceof SequencedSkyframeExecutor);

    UploadContext streamingContext = null;
    Path localOutputFilePath = null;
    String outputFileName;

    if (outputFilePathFragment == null) {
      outputFileName = getDefaultOutputFileName(format);
      if (besOptions != null && besOptions.streamingLogFileUploads) {
        streamingContext =
            runtime
                .getBuildEventArtifactUploaderFactoryMap()
                .select(besOptions.buildEventUploadStrategy)
                .create(env)
                .startUpload(LocalFileType.PERFORMANCE_LOG, /* inputSupplier= */ null);
      } else {
        localOutputFilePath = env.getOutputBase().getRelative(outputFileName);
      }
    } else {
      localOutputFilePath = env.getOutputBase().getRelative(outputFilePathFragment);
      outputFileName = localOutputFilePath.getBaseName();
    }

    if (localOutputFilePath != null) {
      getReporter().handle(Event.info("Writing aquery dump to " + localOutputFilePath));
      getReporter()
          .post(new StartingAqueryDumpAfterBuildEvent(localOutputFilePath, outputFileName));
    } else {
      getReporter().handle(Event.info("Streaming aquery dump."));
      getReporter().post(new StartingAqueryDumpAfterBuildEvent(streamingContext, outputFileName));
    }

    try (OutputStream outputStream = initOutputStream(streamingContext, localOutputFilePath);
        PrintStream printStream = new PrintStream(outputStream)) {
      AqueryOutputHandler aqueryOutputHandler =
          ActionGraphProtoV2OutputFormatterCallback.constructAqueryOutputHandler(
              OutputType.fromString(format), outputStream, printStream);
      // These options are fixed for simplicity. We'll add more configurability if the need arises.
      ActionGraphDump actionGraphDump =
          new ActionGraphDump(
              /* includeActionCmdLine= */ false,
              /* includeArtifacts= */ true,
              /* actionFilters= */ null,
              /* includeParamFiles= */ false,
              aqueryOutputHandler);
      ((SequencedSkyframeExecutor) env.getSkyframeExecutor()).dumpSkyframeState(actionGraphDump);
      aqueryOutputHandler.close();
    }
  }

  private static String getDefaultOutputFileName(String format) {
    switch (format) {
      case "proto":
        return "aquery_dump.proto";
      case "textproto":
        return "aquery_dump.textproto";
      case "jsonproto":
        return "aquery_dump.json";
      default:
        throw new IllegalArgumentException("Unsupported format type: " + format);
    }
  }

  private static OutputStream initOutputStream(
      @Nullable UploadContext streamingContext, Path outputFilePath) throws IOException {
    if (streamingContext != null) {
      return new BufferedOutputStream(streamingContext.getOutputStream());
    }
    return new BufferedOutputStream(outputFilePath.getOutputStream());
  }

  private void reportExceptionError(Exception e) {
    if (e.getMessage() != null) {
      getReporter().handle(Event.error(e.getMessage()));
    }
  }

  /**
   * The crux of the build system. Builds the targets specified in the request using the specified
   * Executor.
   *
   * <p>Performs loading, analysis and execution for the specified set of targets, honoring the
   * configuration options in the BuildRequest. Returns normally iff successful, throws an exception
   * otherwise.
   *
   * <p>The caller is responsible for setting up and syncing the package cache.
   *
   * <p>During this function's execution, the actualTargets and successfulTargets
   * fields of the request object are set.
   *
   * @param request the build request that this build tool is servicing, which specifies various
   *        options; during this method's execution, the actualTargets and successfulTargets fields
   *        of the request object are populated
   * @param validator target validator
   * @return the result as a {@link BuildResult} object
   */
  public BuildResult processRequest(
      BuildRequest request, TargetValidator validator) {
    BuildResult result = new BuildResult(request.getStartTime());
    maybeSetStopOnFirstFailure(request, result);
    int startSuspendCount = suspendCount();
    Throwable catastrophe = null;
    DetailedExitCode detailedExitCode = null;
    try {
      buildTargets(request, result, validator);
      detailedExitCode = DetailedExitCode.success();
    } catch (BuildFailedException e) {
      if (e.isErrorAlreadyShown()) {
        // The actual error has already been reported by the Builder.
      } else {
        reportExceptionError(e);
      }
      if (e.isCatastrophic()) {
        result.setCatastrophe();
      }
      detailedExitCode = e.getDetailedExitCode();
    } catch (InterruptedException e) {
      // We may have been interrupted by an error, or the user's interruption may have raced with
      // an error, so check to see if we should report that error code instead.
      AbruptExitException environmentPendingAbruptExitException = env.getPendingException();
      if (environmentPendingAbruptExitException == null) {
        String message = "build interrupted";
        detailedExitCode = InterruptedFailureDetails.detailedExitCode(message, Code.BUILD);
        env.getReporter().handle(Event.error(message));
        env.getEventBus().post(new BuildInterruptedEvent());
      } else {
        // Report the exception from the environment - the exception we're handling here is just an
        // interruption.
        detailedExitCode = environmentPendingAbruptExitException.getDetailedExitCode();
        reportExceptionError(environmentPendingAbruptExitException);
        result.setCatastrophe();
      }
    } catch (DetailedTargetParsingException e) {
      detailedExitCode = e.getDetailedExitCode();
      reportExceptionError(e);
    } catch (TargetParsingException | LoadingFailedException | ViewCreationFailedException e) {
      detailedExitCode = DetailedExitCode.justExitCode(ExitCode.PARSING_FAILURE);
      reportExceptionError(e);
    } catch (ExitException e) {
      detailedExitCode = e.getDetailedExitCode();
      reportExceptionError(e);
    } catch (TestExecException e) {
      // ExitCode.SUCCESS means that build was successful. Real return code of program
      // is going to be calculated in TestCommand.doTest().
      detailedExitCode = DetailedExitCode.success();
      reportExceptionError(e);
    } catch (InvalidConfigurationException e) {
      detailedExitCode = DetailedExitCode.justExitCode(ExitCode.COMMAND_LINE_ERROR);
      reportExceptionError(e);
      // TODO(gregce): With "global configurations" we cannot tie a configuration creation failure
      // to a single target and have to halt the entire build. Once configurations are genuinely
      // created as part of the analysis phase they should report their error on the level of the
      // target(s) that triggered them.
      result.setCatastrophe();
    } catch (AbruptExitException e) {
      detailedExitCode = e.getDetailedExitCode();
      reportExceptionError(e);
      result.setCatastrophe();
    } catch (PostExecutionActionGraphDumpException e) {
      detailedExitCode =
          DetailedExitCode.of(
              FailureDetail.newBuilder()
                  .setMessage(e.getMessage())
                  .setActionQuery(
                      ActionQuery.newBuilder()
                          .setCode(ActionQuery.Code.SKYFRAME_STATE_AFTER_EXECUTION)
                          .build())
                  .build());
      reportExceptionError(e);
    } catch (Throwable throwable) {
      detailedExitCode = CrashFailureDetails.detailedExitCodeForThrowable(throwable);
      catastrophe = throwable;
      Throwables.propagate(throwable);
    } finally {
      if (detailedExitCode == null) {
        detailedExitCode =
            CrashFailureDetails.detailedExitCodeForThrowable(
                new IllegalStateException("Unspecified DetailedExitCode"));
      }
      stopRequest(result, catastrophe, detailedExitCode, startSuspendCount);
    }

    return result;
  }

  private void maybeSetStopOnFirstFailure(BuildRequest request, BuildResult result) {
    if (shouldStopOnFailure(request)) {
      result.setStopOnFirstFailure(true);
    }
  }

  private boolean shouldStopOnFailure(BuildRequest request) {
    return !(request.getKeepGoing() && request.getExecutionOptions().testKeepGoing);
  }

  /**
   * Initializes the output filter to the value given with {@code --output_filter}.
   */
  private void initializeOutputFilter(BuildRequest request) {
    RegexPatternOption outputFilterOption = request.getBuildOptions().outputFilter;
    if (outputFilterOption != null) {
      getReporter()
          .setOutputFilter(
              OutputFilter.RegexOutputFilter.forPattern(outputFilterOption.regexPattern()));
    }
  }

  private static boolean needsExecutionPhase(BuildRequestOptions options) {
    return options.performAnalysisPhase && options.performExecutionPhase;
  }

  /**
   * Stops processing the specified request.
   *
   * <p>This logs the build result, cleans up and stops the clock.
   *
   * @param result result to update
   * @param crash any unexpected {@link RuntimeException} or {@link Error}. May be null
   * @param detailedExitCode describes the exit code and an optional detailed failure value to add
   *     to {@code result}
   * @param startSuspendCount number of suspensions before the build started
   */
  public void stopRequest(
      BuildResult result,
      Throwable crash,
      DetailedExitCode detailedExitCode,
      int startSuspendCount) {
    Preconditions.checkState((crash == null) || !detailedExitCode.isSuccess());
    int stopSuspendCount = suspendCount();
    Preconditions.checkState(startSuspendCount <= stopSuspendCount);
    result.setUnhandledThrowable(crash);
    result.setDetailedExitCode(detailedExitCode);
    InterruptedException ie = null;
    try {
      env.getSkyframeExecutor().notifyCommandComplete(env.getReporter());
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("Build interrupted during command completion"));
      ie = e;
    }
    // The stop time has to be captured before we send the BuildCompleteEvent.
    result.setStopTime(runtime.getClock().currentTimeMillis());
    result.setWasSuspended(stopSuspendCount > startSuspendCount);

    env.getEventBus().post(new BuildPrecompleteEvent());
    env.getEventBus()
        .post(
            new BuildCompleteEvent(
                result,
                ImmutableList.of(
                    BuildEventIdUtil.buildToolLogs(), BuildEventIdUtil.buildMetrics())));
    // Post the build tool logs event; the corresponding local files may be contributed from
    // modules, and this has to happen after posting the BuildCompleteEvent because that's when
    // modules add their data to the collection.
    env.getEventBus().post(result.getBuildToolLogCollection().freeze().toEvent());
    if (ie != null) {
      if (detailedExitCode.isSuccess()) {
        result.setDetailedExitCode(
            InterruptedFailureDetails.detailedExitCode(
                "Build interrupted during command completion", Code.BUILD_COMPLETION));
      } else if (!detailedExitCode.getExitCode().equals(ExitCode.INTERRUPTED)) {
        logger.atWarning().withCause(ie).log(
            "Suppressed interrupted exception during stop request because already failing with: %s",
            detailedExitCode);
      }
    }
  }

  /**
   * Validates the options for this BuildRequest.
   *
   * <p>Issues warnings for the use of deprecated options, and warnings or errors for any option
   * settings that conflict.
   */
  @VisibleForTesting
  public void validateOptions(BuildRequest request) throws InvalidConfigurationException {
    for (String issue : request.validateOptions()) {
      getReporter().handle(Event.warn(issue));
    }
  }

  private Reporter getReporter() {
    return env.getReporter();
  }

  /** Describes a failure that isn't severe enough to halt the command in keep_going mode. */
  // TODO(mschaller): consider promoting this to be a sibling of AbruptExitException.
  static class ExitException extends Exception {

    private final DetailedExitCode detailedExitCode;

    ExitException(DetailedExitCode detailedExitCode) {
      super(
          Preconditions.checkNotNull(detailedExitCode.getFailureDetail(), "failure detail")
              .getMessage());
      this.detailedExitCode = detailedExitCode;
    }

    DetailedExitCode getDetailedExitCode() {
      return detailedExitCode;
    }
  }
}
