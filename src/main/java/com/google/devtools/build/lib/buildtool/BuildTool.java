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

import static java.util.stream.Collectors.joining;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Stopwatch;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMultimap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.LinkedListMultimap;
import com.google.common.collect.ListMultimap;
import com.google.common.collect.Multimaps;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.CommandLineExpansionException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.AnalysisAndExecutionResult;
import com.google.devtools.build.lib.analysis.AnalysisResult;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.Project;
import com.google.devtools.build.lib.analysis.Project.ProjectParseException;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.actions.TemplateExpansionException;
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
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.TargetParsingException;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.OutputFilter;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.exec.ExecutionOptions;
import com.google.devtools.build.lib.pkgcache.LoadingFailedException;
import com.google.devtools.build.lib.profiler.ProfilePhase;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.query2.aquery.ActionGraphProtoOutputFormatterCallback;
import com.google.devtools.build.lib.runtime.BlazeRuntime;
import com.google.devtools.build.lib.runtime.CommandEnvironment;
import com.google.devtools.build.lib.server.FailureDetails.ActionQuery;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.BuildResultListener;
import com.google.devtools.build.lib.skyframe.RepositoryMappingValue.RepositoryMappingResolutionException;
import com.google.devtools.build.lib.skyframe.SequencedSkyframeExecutor;
import com.google.devtools.build.lib.skyframe.SkyframeBuildView.BuildDriverKeyTestContext;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TargetPatternPhaseValue;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.ActionGraphDump;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.AqueryOutputHandler.OutputType;
import com.google.devtools.build.lib.skyframe.actiongraph.v2.InvalidAqueryOutputFormatException;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.CrashFailureDetails;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.InterruptedFailureDetails;
import com.google.devtools.build.lib.vfs.Path;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.common.options.RegexPatternOption;
import java.io.BufferedOutputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.util.Collection;
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

  private static final AnalysisPostProcessor NOOP_POST_PROCESSOR =
      (req, env, runtime, analysisResult) -> {};

  /** Hook for inserting extra post-analysis-phase processing. Used for implementing {a,c}query. */
  public interface AnalysisPostProcessor {
    void process(
        BuildRequest request,
        CommandEnvironment env,
        BlazeRuntime runtime,
        AnalysisResult analysisResult)
        throws InterruptedException, ViewCreationFailedException, ExitException;
  }

  private final CommandEnvironment env;
  private final BlazeRuntime runtime;
  private final AnalysisPostProcessor analysisPostProcessor;

  /**
   * Constructs a BuildTool.
   *
   * @param env a reference to the command environment of the currently executing command
   */
  public BuildTool(CommandEnvironment env) {
    this(env, NOOP_POST_PROCESSOR);
  }

  public BuildTool(CommandEnvironment env, AnalysisPostProcessor postProcessor) {
    this.env = env;
    this.runtime = env.getRuntime();
    this.analysisPostProcessor = postProcessor;
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
          PostExecutionActionGraphDumpException, RepositoryMappingResolutionException {
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
        env.getEventBus().post(BuildStartingEvent.create(env, request));
      }
      logger.atInfo().log("Build identifier: %s", request.getId());

      // Exit if there are any pending exceptions from modules.
      env.throwPendingException();

      initializeOutputFilter(request);

      if (env.withMergedAnalysisAndExecutionSourceOfTruth()) {
        buildTargetsWithMergedAnalysisExecution(request, result, validator, buildOptions);
        return;
      }

      AnalysisResult analysisResult =
          AnalysisPhaseRunner.execute(env, request, buildOptions, validator);

      // We cannot move the executionTool down to the execution phase part since it does set up the
      // symlinks for tools.
      // TODO(twerth): Extract embedded tool setup from execution tool and move object creation to
      // execution phase.
      executionTool = new ExecutionTool(env, request);
      if (request.getBuildOptions().performAnalysisPhase) {

        if (!analysisResult.getExclusiveTests().isEmpty()
            && executionTool.getTestActionContext().forceExclusiveTestsInParallel()) {
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
        if (!analysisResult.getExclusiveIfLocalTests().isEmpty()
            && executionTool.getTestActionContext().forceExclusiveIfLocalTestsInParallel()) {
          analysisResult = analysisResult.withExclusiveIfLocalTestsAsParallelTests();
        }

        result.setBuildConfiguration(analysisResult.getConfiguration());
        result.setActualTargets(analysisResult.getTargetsToBuild());
        result.setTestTargets(analysisResult.getTargetsToTest());

        try (SilentCloseable c = Profiler.instance().profile("analysisPostProcessor.process")) {
          analysisPostProcessor.process(request, env, runtime, analysisResult);
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
          } catch (CommandLineExpansionException | IOException | TemplateExpansionException e) {
            throw new PostExecutionActionGraphDumpException(e);
          } catch (InvalidAqueryOutputFormatException e) {
            throw new PostExecutionActionGraphDumpException(
                "--skyframe_state must be used with "
                    + "--output=proto|streamed_proto|textproto|jsonproto.",
                e);
          }
        }
      }
    } catch (Error | RuntimeException e) {
      // Don't handle the error here. We will do so in stopRequest.
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
        // The workspace status actions will not run with certain flags, or if an error occurs early
        // in the build. Ensure that build info is posted on every build.
        env.ensureBuildInfoPosted();
      }
    }
  }

  /** Performs the merged analysis and execution phase. */
  private void buildTargetsWithMergedAnalysisExecution(
      BuildRequest request,
      BuildResult result,
      TargetValidator validator,
      BuildOptions buildOptions)
      throws InterruptedException, TargetParsingException, LoadingFailedException,
          AbruptExitException, ViewCreationFailedException, BuildFailedException, TestExecException,
          InvalidConfigurationException, RepositoryMappingResolutionException {
    // Target pattern evaluation.
    TargetPatternPhaseValue loadingResult;
    Profiler.instance().markPhase(ProfilePhase.TARGET_PATTERN_EVAL);
    try (SilentCloseable c = Profiler.instance().profile("evaluateTargetPatterns")) {
      loadingResult =
          AnalysisAndExecutionPhaseRunner.evaluateTargetPatterns(env, request, validator);
    }
    env.setWorkspaceName(loadingResult.getWorkspaceName());

    // TODO: b/324127375 - Use with incoming --scl_config flag.
    PathFragment unusedProjectFile =
        getProjectFile(loadingResult.getTargetLabels(), env.getSkyframeExecutor(), getReporter());

    // See https://github.com/bazelbuild/rules_nodejs/issues/3693.
    env.getSkyframeExecutor().clearSyscallCache();

    boolean hasCatastrophe = false;

    ExecutionTool executionTool = new ExecutionTool(env, request);
    // This timer measures time from the first execution activity to the last.
    Stopwatch executionTimer = Stopwatch.createUnstarted();

    // TODO(b/199053098): implement support for --nobuild.
    AnalysisAndExecutionResult analysisAndExecutionResult = null;
    boolean buildCompleted = false;
    try {
      analysisAndExecutionResult =
          AnalysisAndExecutionPhaseRunner.execute(
              env,
              request,
              buildOptions,
              loadingResult,
              () -> executionTool.prepareForExecution(executionTimer),
              result::setBuildConfiguration,
              new BuildDriverKeyTestContext() {
                @Override
                public String getTestStrategy() {
                  return request.getOptions(ExecutionOptions.class).testStrategy;
                }

                @Override
                public boolean forceExclusiveTestsInParallel() {
                  return executionTool.getTestActionContext().forceExclusiveTestsInParallel();
                }

                @Override
                public boolean forceExclusiveIfLocalTestsInParallel() {
                  return executionTool
                      .getTestActionContext()
                      .forceExclusiveIfLocalTestsInParallel();
                }
              });
      buildCompleted = true;

      // This value is null when there's no analysis.
      if (analysisAndExecutionResult == null) {
        return;
      }
    } catch (InvalidConfigurationException
        | RepositoryMappingResolutionException
        | ViewCreationFailedException
        | BuildFailedException
        | TestExecException e) {
      // These are non-catastrophic.
      buildCompleted = true;
      throw e;
    } catch (Error | RuntimeException e) {
      // These are catastrophic.
      hasCatastrophe = true;
      throw e;
    } finally {
      if (result.getBuildConfiguration() != null) {
        // We still need to do this even in case of an exception.
        result.setConvenienceSymlinks(
            executionTool.handleConvenienceSymlinks(
                env.getBuildResultListener().getAnalyzedTargets(), result.getBuildConfiguration()));
      }
      executionTool.unconditionalExecutionPhaseFinalizations(
          executionTimer, env.getSkyframeExecutor());

      // For the --noskymeld code path, this is done after the analysis phase.
      BuildResultListener buildResultListener = env.getBuildResultListener();
      result.setActualTargets(buildResultListener.getAnalyzedTargets());
      result.setTestTargets(buildResultListener.getAnalyzedTests());

      if (!hasCatastrophe) {
        executionTool.nonCatastrophicFinalizations(
            result,
            env.getBlazeWorkspace().getInUseActionCacheWithoutFurtherLoading(),
            /* explanationHandler= */ null,
            buildCompleted);
      }
    }

    // This is the --keep_going code path: Time to throw the delayed exceptions.
    // Keeping legacy behavior: for execution errors, keep the message of the BuildFailedException
    // empty.
    if (analysisAndExecutionResult.getExecutionDetailedExitCode() != null) {
      throw new BuildFailedException(
          null, analysisAndExecutionResult.getExecutionDetailedExitCode());
    }

    FailureDetail delayedFailureDetail = analysisAndExecutionResult.getFailureDetail();
    if (delayedFailureDetail != null) {
      throw new BuildFailedException(
          delayedFailureDetail.getMessage(), DetailedExitCode.of(delayedFailureDetail));
    }
  }

  /**
   * Produces an aquery dump of the state of Skyframe.
   *
   * <p>There are 2 possible output channels: a local file or a remote FS.
   */
  private void dumpSkyframeStateAfterBuild(
      @Nullable BuildEventProtocolOptions besOptions,
      String format,
      @Nullable PathFragment outputFilePathFragment)
      throws CommandLineExpansionException, IOException, InvalidAqueryOutputFormatException,
          TemplateExpansionException {
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
        PrintStream printStream = new PrintStream(outputStream);
        AqueryOutputHandler aqueryOutputHandler =
            ActionGraphProtoOutputFormatterCallback.constructAqueryOutputHandler(
                OutputType.fromString(format), outputStream, printStream)) {
      // These options are fixed for simplicity. We'll add more configurability if the need arises.
      ActionGraphDump actionGraphDump =
          new ActionGraphDump(
              /* includeActionCmdLine= */ false,
              /* includeArtifacts= */ true,
              /* includeSchedulingDependencies= */ true,
              /* actionFilters= */ null,
              /* includeParamFiles= */ false,
              /* includeFileWriteContents */ false,
              aqueryOutputHandler,
              getReporter());
      AqueryProcessor.dumpActionGraph(env, aqueryOutputHandler, actionGraphDump);
    }
  }

  private static String getDefaultOutputFileName(String format) {
    switch (format) {
      case "proto":
        return "aquery_dump.proto";
      case "streamed_proto":
        return "aquery_dump.pb";
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

  public BuildResult processRequest(BuildRequest request, TargetValidator validator) {
    return processRequest(request, validator, /* postBuildCallback= */ null);
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
   * <p>During this function's execution, the actualTargets and successfulTargets fields of the
   * request object are set.
   *
   * @param request the build request that this build tool is servicing, which specifies various
   *     options; during this method's execution, the actualTargets and successfulTargets fields of
   *     the request object are populated
   * @param validator an optional target validator
   * @param postBuildCallback an optional callback called after the build has been completed
   *     successfully
   * @return the result as a {@link BuildResult} object
   */
  public BuildResult processRequest(
      BuildRequest request, TargetValidator validator, PostBuildCallback postBuildCallback) {
    BuildResult result = new BuildResult(request.getStartTime());
    maybeSetStopOnFirstFailure(request, result);
    Throwable crash = null;
    DetailedExitCode detailedExitCode = null;
    try {
      try (SilentCloseable c = Profiler.instance().profile("buildTargets")) {
        buildTargets(request, result, validator);
      }
      detailedExitCode = DetailedExitCode.success();
      if (postBuildCallback != null) {
        try (SilentCloseable c = Profiler.instance().profile("postBuildCallback.process")) {
          result.setPostBuildCallbackFailureDetail(
              postBuildCallback.process(result.getSuccessfulTargets()));
        } catch (InterruptedException e) {
          detailedExitCode =
              InterruptedFailureDetails.detailedExitCode("post build callback interrupted");
        }
      }
    } catch (BuildFailedException e) {
      if (!e.isErrorAlreadyShown()) {
        // The actual error has not already been reported by the Builder.
        // TODO(janakr): This is wrong: --keep_going builds with errors don't have a message in
        //  this BuildFailedException, so any error message that is only reported here will be
        //  missing for --keep_going builds. All error reporting should be done at the site of the
        //  error, if only for clearer behavior.
        reportExceptionError(e);
      }
      if (e.isCatastrophic()) {
        result.setCatastrophe();
      }
      detailedExitCode = e.getDetailedExitCode();
    } catch (InterruptedException e) {
      // We may have been interrupted by an error, or the user's interruption may have raced with
      // an error, so check to see if we should report that error code instead.
      detailedExitCode = env.getRuntime().getCrashExitCode();
      AbruptExitException environmentPendingAbruptExitException = env.getPendingException();
      if (detailedExitCode == null && environmentPendingAbruptExitException != null) {
        detailedExitCode = environmentPendingAbruptExitException.getDetailedExitCode();
        // Report the exception from the environment - the exception we're handling here is just an
        // interruption.
        reportExceptionError(environmentPendingAbruptExitException);
      }
      if (detailedExitCode == null) {
        String message = "build interrupted";
        detailedExitCode = InterruptedFailureDetails.detailedExitCode(message);
        env.getReporter().handle(Event.error(message));
        env.getEventBus().post(new BuildInterruptedEvent());
      } else {
        result.setCatastrophe();
      }
    } catch (TargetParsingException | LoadingFailedException e) {
      detailedExitCode = e.getDetailedExitCode();
      reportExceptionError(e);
    } catch (RepositoryMappingResolutionException e) {
      detailedExitCode = e.getDetailedExitCode();
      reportExceptionError(e);
    } catch (ViewCreationFailedException e) {
      detailedExitCode = DetailedExitCode.of(ExitCode.PARSING_FAILURE, e.getFailureDetail());
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
      detailedExitCode = e.getDetailedExitCode();
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
      crash = throwable;
      detailedExitCode = CrashFailureDetails.detailedExitCodeForThrowable(crash);
      Throwables.throwIfUnchecked(throwable);
      throw new IllegalStateException(throwable);
    } finally {
      if (detailedExitCode == null) {
        detailedExitCode =
            CrashFailureDetails.detailedExitCodeForThrowable(
                new IllegalStateException("Unspecified DetailedExitCode"));
      }
      try (SilentCloseable c = Profiler.instance().profile("stopRequest")) {
        stopRequest(result, crash, detailedExitCode);
      }
    }

    return result;
  }

  private static void maybeSetStopOnFirstFailure(BuildRequest request, BuildResult result) {
    if (shouldStopOnFailure(request)) {
      result.setStopOnFirstFailure(true);
    }
  }

  private static boolean shouldStopOnFailure(BuildRequest request) {
    return !(request.getKeepGoing() && request.getExecutionOptions().testKeepGoing);
  }

  /** Initializes the output filter to the value given with {@code --output_filter}. */
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
   * @param crash any unexpected {@link RuntimeException} or {@link Error}, may be null
   * @param detailedExitCode describes the exit code and an optional detailed failure value to add
   *     to {@code result}
   */
  public void stopRequest(
      BuildResult result, @Nullable Throwable crash, DetailedExitCode detailedExitCode) {
    Preconditions.checkState((crash == null) || !detailedExitCode.isSuccess());
    result.setUnhandledThrowable(crash);
    result.setDetailedExitCode(detailedExitCode);
    if (!detailedExitCode.isSuccess()) {
      logger.atInfo().log(
          "Unsuccessful command ended with FailureDetail: %s", detailedExitCode.getFailureDetail());
    }

    InterruptedException ie = null;
    try {
      env.getSkyframeExecutor().notifyCommandComplete(env.getReporter());
    } catch (InterruptedException e) {
      env.getReporter().handle(Event.error("Build interrupted during command completion"));
      ie = e;
    }
    // Handle subscribers that need to run between the end of the command, and the end of the
    // invocation proper. Tasks here will count to the overall time in profiles/metrics, unlike
    // BuildPrecompleteEvent below.
    env.getEventBus().post(new BuildToolFinalizingEvent(result.getDetailedExitCode()));

    // The stop time has to be captured before we send the BuildCompleteEvent.
    result.setStopTime(runtime.getClock().currentTimeMillis());

    // Skip the build complete events so that modules can run blazeShutdownOnCrash without thinking
    // that the build completed normally. BlazeCommandDispatcher will call handleCrash.
    if (crash == null) {
      try {
        Profiler.instance().markPhase(ProfilePhase.FINISH);
      } catch (InterruptedException e) {
        env.getReporter().handle(Event.error("Build interrupted during command completion"));
        ie = e;
      }
      // Ensure deterministic ordering of doing the metrics upload before everything else that
      // happens when BuildCompleteEvent is posted.
      env.getEventBus().post(new BuildPrecompleteEvent());
      env.getEventBus()
          .post(
              new BuildCompleteEvent(
                  result,
                  ImmutableList.of(
                      BuildEventIdUtil.buildToolLogs(), BuildEventIdUtil.buildMetrics())));
    }
    // Post the build tool logs event; the corresponding local files may be contributed from
    // modules, and this has to happen after posting the BuildCompleteEvent because that's when
    // modules add their data to the collection.
    env.getEventBus().post(result.getBuildToolLogCollection().freeze().toEvent());
    if (ie != null) {
      if (detailedExitCode.isSuccess()) {
        result.setDetailedExitCode(
            InterruptedFailureDetails.detailedExitCode(
                "Build interrupted during command completion"));
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
  public void validateOptions(BuildRequest request) {
    for (String issue : request.validateOptions()) {
      getReporter().handle(Event.warn(issue));
    }
  }

  /**
   * Returns the project file suitable for this build, or null if there's no project file.
   *
   * @param topLevelTargets this build's top-level targets
   * @param skyframeExecutor support for SkyFunctions that look up project files
   * @param eventHandler event handler
   * @throws LoadingFailedException if this build doesn't cleanly resolve to a single project file.
   *     Builds are valid if either a) all top-level targets resolve to the same project file or b)
   *     none of the top-level targets resolve to any project file. Builds are invalid if a) any
   *     top-level targets resolve to different project files or b) any top-level target has
   *     multiple project files (i.e. foo/project.scl and foo/bar/project.scl).
   */
  // TODO: b/324127375 - Support hierarchical project files: [foo/project.scl, foo/bar/project.scl].
  @Nullable
  @VisibleForTesting
  static PathFragment getProjectFile(
      Collection<Label> topLevelTargets,
      SkyframeExecutor skyframeExecutor,
      ExtendedEventHandler eventHandler)
      throws LoadingFailedException {
    ImmutableMultimap<Label, PathFragment> projectFiles;
    try {
      projectFiles = Project.findProjectFiles(topLevelTargets, skyframeExecutor, eventHandler);
    } catch (ProjectParseException e) {
      throw new LoadingFailedException(
          "Error finding project files: " + e.getMessage(),
          DetailedExitCode.of(ExitCode.PARSING_FAILURE, FailureDetail.getDefaultInstance()));
    }

    ImmutableSet<PathFragment> distinct = ImmutableSet.copyOf(projectFiles.values());
    if (distinct.size() == 1) {
      PathFragment projectFile = Iterables.getOnlyElement(distinct);
      eventHandler.handle(
          Event.info(String.format("Reading project settings from %s.", projectFile)));
      return projectFile;
    } else if (distinct.isEmpty()) {
      return null;
    } else {
      ListMultimap<Collection<PathFragment>, Label> projectFilesToTargets =
          LinkedListMultimap.create();
      Multimaps.invertFrom(Multimaps.forMap(projectFiles.asMap()), projectFilesToTargets);
      StringBuilder msgBuilder =
          new StringBuilder("This build doesn't support automatic project resolution. ");
      if (projectFilesToTargets.size() == 1) {
        msgBuilder
            .append("Multiple project files found: ")
            .append(
                projectFilesToTargets.keys().stream().map(Object::toString).collect(joining(",")));
      } else {
        msgBuilder.append("Targets have different project settings. For example: ");
        for (var entry : projectFilesToTargets.asMap().entrySet()) {
          msgBuilder.append(
              String.format(
                  " [%s]: %s",
                  entry.getKey().stream().map(l -> l.toString()).collect(joining(",")),
                  entry.getValue().iterator().next()));
        }
      }
      String errorMsg = msgBuilder.toString();
      throw new LoadingFailedException(
          errorMsg,
          DetailedExitCode.of(ExitCode.BUILD_FAILURE, FailureDetail.getDefaultInstance()));
    }
  }

  private Reporter getReporter() {
    return env.getReporter();
  }

  /** Describes a failure that isn't severe enough to halt the command in keep_going mode. */
  // TODO(mschaller): consider promoting this to be a sibling of AbruptExitException.
  public static class ExitException extends Exception {

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
