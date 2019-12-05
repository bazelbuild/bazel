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

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Optional;
import com.google.common.base.Preconditions;
import com.google.common.collect.Range;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInputPrefetcher;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MetadataProvider;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.analysis.TopLevelArtifactContext;
import com.google.devtools.build.lib.analysis.test.TestProvider;
import com.google.devtools.build.lib.buildtool.buildevent.ExecutionProgressReceiverAvailableEvent;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.runtime.KeepGoingOption;
import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.AspectValue.AspectKey;
import com.google.devtools.build.lib.skyframe.Builder;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.skyframe.TopDownActionCache;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.ExitCode;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.common.options.OptionsProvider;
import java.util.Collection;
import java.util.Collections;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.logging.Level;
import javax.annotation.Nullable;

/**
 * A {@link Builder} implementation driven by Skyframe.
 */
@VisibleForTesting
public class SkyframeBuilder implements Builder {

  private final ResourceManager resourceManager;
  private final SkyframeExecutor skyframeExecutor;
  private final ModifiedFileSet modifiedOutputFiles;
  private final MetadataProvider fileCache;
  private final ActionInputPrefetcher actionInputPrefetcher;
  private final ActionCacheChecker actionCacheChecker;
  private final TopDownActionCache topDownActionCache;

  @VisibleForTesting
  public SkyframeBuilder(
      SkyframeExecutor skyframeExecutor,
      ResourceManager resourceManager,
      ActionCacheChecker actionCacheChecker,
      TopDownActionCache topDownActionCache,
      ModifiedFileSet modifiedOutputFiles,
      MetadataProvider fileCache,
      ActionInputPrefetcher actionInputPrefetcher) {
    this.resourceManager = resourceManager;
    this.skyframeExecutor = skyframeExecutor;
    this.actionCacheChecker = actionCacheChecker;
    this.topDownActionCache = topDownActionCache;
    this.modifiedOutputFiles = modifiedOutputFiles;
    this.fileCache = fileCache;
    this.actionInputPrefetcher = actionInputPrefetcher;
  }

  @Override
  public void buildArtifacts(
      Reporter reporter,
      Set<Artifact> artifacts,
      Set<ConfiguredTarget> parallelTests,
      Set<ConfiguredTarget> exclusiveTests,
      Set<ConfiguredTarget> targetsToBuild,
      Set<ConfiguredTarget> targetsToSkip,
      Collection<AspectValue> aspects,
      Executor executor,
      Set<ConfiguredTargetKey> builtTargets,
      Set<AspectKey> builtAspects,
      OptionsProvider options,
      @Nullable Range<Long> lastExecutionTimeRange,
      TopLevelArtifactContext topLevelArtifactContext)
      throws BuildFailedException, AbruptExitException, TestExecException, InterruptedException {
    skyframeExecutor.detectModifiedOutputFiles(modifiedOutputFiles, lastExecutionTimeRange);
    try (SilentCloseable c = Profiler.instance().profile("configureActionExecutor")) {
      skyframeExecutor.configureActionExecutor(fileCache, actionInputPrefetcher);
    }
    // Note that executionProgressReceiver accesses builtTargets concurrently (after wrapping in a
    // synchronized collection), so unsynchronized access to this variable is unsafe while it runs.
    ExecutionProgressReceiver executionProgressReceiver =
        new ExecutionProgressReceiver(
            Preconditions.checkNotNull(builtTargets),
            Preconditions.checkNotNull(builtAspects),
            countTestActions(exclusiveTests));
    skyframeExecutor
        .getEventBus()
        .post(new ExecutionProgressReceiverAvailableEvent(executionProgressReceiver));

    List<ExitCode> exitCodes = new LinkedList<>();
    EvaluationResult<?> result;

    ActionExecutionStatusReporter statusReporter = ActionExecutionStatusReporter.create(
        reporter, skyframeExecutor.getEventBus());

    AtomicBoolean isBuildingExclusiveArtifacts = new AtomicBoolean(false);
    ActionExecutionInactivityWatchdog watchdog =
        new ActionExecutionInactivityWatchdog(
            executionProgressReceiver.createInactivityMonitor(statusReporter),
            executionProgressReceiver.createInactivityReporter(
                statusReporter, isBuildingExclusiveArtifacts),
            options.getOptions(BuildRequestOptions.class).progressReportInterval);

    skyframeExecutor.setActionExecutionProgressReportingObjects(executionProgressReceiver,
        executionProgressReceiver, statusReporter);
    watchdog.start();

    targetsToBuild = Sets.difference(targetsToBuild, targetsToSkip);
    parallelTests = Sets.difference(parallelTests, targetsToSkip);
    exclusiveTests = Sets.difference(exclusiveTests, targetsToSkip);

    try {
      result =
          skyframeExecutor.buildArtifacts(
              reporter,
              resourceManager,
              executor,
              artifacts,
              targetsToBuild,
              aspects,
              parallelTests,
              exclusiveTests,
              options,
              actionCacheChecker,
              topDownActionCache,
              executionProgressReceiver,
              topLevelArtifactContext);
      // progressReceiver is finished, so unsynchronized access to builtTargets is now safe.
      Optional<ExitCode> exitCode =
          processResult(
              reporter,
              result,
              options.getOptions(KeepGoingOption.class).keepGoing,
              skyframeExecutor);

      if (exitCode != null) {
        exitCodes.add(exitCode.orNull());
      }

      // Run exclusive tests: either tagged as "exclusive" or is run in an invocation with
      // --test_output=streamed.
      isBuildingExclusiveArtifacts.set(true);
      for (ConfiguredTarget exclusiveTest : exclusiveTests) {
        // Since only one artifact is being built at a time, we don't worry about an artifact being
        // built and then the build being interrupted.
        result =
            skyframeExecutor.runExclusiveTest(
                reporter,
                resourceManager,
                executor,
                exclusiveTest,
                options,
                actionCacheChecker,
                topDownActionCache,
                null,
                topLevelArtifactContext);
        exitCode =
            processResult(
                reporter,
                result,
                options.getOptions(KeepGoingOption.class).keepGoing,
                skyframeExecutor);
        Preconditions.checkState(
            exitCode != null || !result.keyNames().isEmpty(),
            "Build reported as successful but test %s not executed: %s",
            exclusiveTest,
            result);

        if (exitCode != null) {
          exitCodes.add(exitCode.orNull());
        }
      }
    } finally {
      watchdog.stop();
      skyframeExecutor.setActionExecutionProgressReportingObjects(null, null, null);
      statusReporter.unregisterFromEventBus();
    }

    if (!exitCodes.isEmpty()) {
      if (options.getOptions(KeepGoingOption.class).keepGoing) {
        // Use the exit code with the highest priority.
        throw new BuildFailedException(
            null, Collections.max(exitCodes, ExitCodeComparator.INSTANCE));
      } else {
        throw new BuildFailedException();
      }
    }
  }

  /**
   * Process the Skyframe update, taking into account the keepGoing setting.
   *
   * <p>Returns optional {@link ExitCode} based on following conditions: 1. null, if result had no
   * errors. 2. Optional.absent(), if result had errors but none of the errors specified an exit
   * code. 3. Optional.of(e), if result had errors and one of them specified exit code 'e'. Throws
   * on fail-fast failures.
   */
  @Nullable
  private static Optional<ExitCode> processResult(
      ExtendedEventHandler eventHandler,
      EvaluationResult<?> result,
      boolean keepGoing,
      SkyframeExecutor skyframeExecutor)
      throws BuildFailedException, TestExecException {
    if (result.hasError()) {
      for (Map.Entry<SkyKey, ErrorInfo> entry : result.errorMap().entrySet()) {
        Iterable<CycleInfo> cycles = entry.getValue().getCycleInfo();
        skyframeExecutor.reportCycles(eventHandler, cycles, entry.getKey());
      }

      if (result.getCatastrophe() != null) {
        rethrow(result.getCatastrophe());
      }
      if (keepGoing) {
        // If build fails and keepGoing is true, an exit code is assigned using reported errors
        // in the following order:
        //   1. First infrastructure error with non-null exit code
        //   2. First non-infrastructure error with non-null exit code
        //   3. Null (later default to 1)
        ExitCode exitCode = null;
        for (Map.Entry<SkyKey, ErrorInfo> error : result.errorMap().entrySet()) {
          Throwable cause = error.getValue().getException();
          if (cause instanceof ActionExecutionException) {
            ActionExecutionException actionExecutionCause = (ActionExecutionException) cause;
            ExitCode code = actionExecutionCause.getExitCode();
            // Update global exit code when current exit code is not null and global exit code has
            // a lower 'reporting' priority.
            if (ExitCodeComparator.INSTANCE.compare(code, exitCode) > 0) {
              exitCode = code;
            }
          }
        }

        return Optional.fromNullable(exitCode);
      }
      ErrorInfo errorInfo = Preconditions.checkNotNull(result.getError(), result);
      Exception exception = errorInfo.getException();
      if (exception == null) {
        Preconditions.checkState(!errorInfo.getCycleInfo().isEmpty(), errorInfo);
        // If a keepGoing=false build found a cycle, that means there were no other errors thrown
        // during evaluation (otherwise, it wouldn't have bothered to find a cycle). So the best
        // we can do is throw a generic build failure exception, since we've already reported the
        // cycles above.
        throw new BuildFailedException(null, /* catastrophic= */ false);
      } else {
        rethrow(exception);
      }
    }

    return null;
  }

  /** Figure out why an action's execution failed and rethrow the right kind of exception. */
  @VisibleForTesting
  public static void rethrow(Throwable cause) throws BuildFailedException, TestExecException {
    Throwable innerCause = cause.getCause();
    if (innerCause instanceof TestExecException) {
      throw (TestExecException) innerCause;
    }
    if (cause instanceof ActionExecutionException) {
      ActionExecutionException actionExecutionCause = (ActionExecutionException) cause;
      // Sometimes ActionExecutionExceptions are caused by Actions with no owner.
      String message =
          (actionExecutionCause.getLocation() != null)
              ? (actionExecutionCause.getLocation().print() + " " + cause.getMessage())
              : cause.getMessage();
      throw new BuildFailedException(
          message,
          actionExecutionCause.isCatastrophe(),
          actionExecutionCause.getAction(),
          actionExecutionCause.getRootCauses(),
          /*errorAlreadyShown=*/ !actionExecutionCause.showError(),
          actionExecutionCause.getExitCode());
    } else if (cause instanceof MissingInputFileException) {
      throw new BuildFailedException(cause.getMessage());
    } else if (cause instanceof BuildFileNotFoundException) {
      // Sadly, this can happen because we may load new packages during input discovery. Any
      // failures reading those packages shouldn't terminate the build, but in Skyframe they do.
      LoggingUtil.logToRemote(Level.WARNING, "undesirable loading exception", cause);
      throw new BuildFailedException(cause.getMessage());
    } else if (cause instanceof RuntimeException) {
      throw (RuntimeException) cause;
    } else if (cause instanceof Error) {
      throw (Error) cause;
    } else {
      // We encountered an exception we don't think we should have encountered. This can indicate
      // a bug in our code, such as lower level exceptions not being properly handled, or in our
      // expectations in this method.
      throw new IllegalArgumentException(
          "action terminated with " + "unexpected exception: " + cause.getMessage(), cause);
    }
  }

  private static int countTestActions(Iterable<ConfiguredTarget> testTargets) {
    int count = 0;
    for (ConfiguredTarget testTarget : testTargets) {
      count += TestProvider.getTestStatusArtifacts(testTarget).size();
    }
    return count;
  }

  /**
   * A comparator to determine the reporting priority of {@link ExitCode}.
   *
   * <p> Priority: infrastructure exit codes > non-infrastructure exit codes > null exit codes.
   */
  private static class ExitCodeComparator implements Comparator<ExitCode> {
    private static final ExitCodeComparator INSTANCE = new ExitCodeComparator();

    @Override
    public int compare(ExitCode c1, ExitCode c2) {
      // returns POSITIVE result when the priority of c1 is HIGHER than the priority of c2
      return getPriority(c1) - getPriority(c2);
    }

    private int getPriority(ExitCode code) {
      if (code == null) {
        return 0;
      } else {
        return code.isInfrastructureFailure() ? 2 : 1;
      }
    }
  }
}
