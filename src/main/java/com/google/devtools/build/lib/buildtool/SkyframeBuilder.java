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
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Range;
import com.google.devtools.build.lib.actions.ActionCacheChecker;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionExecutionStatusReporter;
import com.google.devtools.build.lib.actions.ActionInputFileCache;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.Executor;
import com.google.devtools.build.lib.actions.MissingInputFileException;
import com.google.devtools.build.lib.actions.ResourceManager;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.events.Reporter;
import com.google.devtools.build.lib.packages.BuildFileNotFoundException;
import com.google.devtools.build.lib.rules.test.TestProvider;
import com.google.devtools.build.lib.skyframe.ActionExecutionInactivityWatchdog;
import com.google.devtools.build.lib.skyframe.AspectValue;
import com.google.devtools.build.lib.skyframe.Builder;
import com.google.devtools.build.lib.skyframe.SkyframeExecutor;
import com.google.devtools.build.lib.util.AbruptExitException;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.lib.vfs.ModifiedFileSet;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;

import java.util.Collection;
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

  private final SkyframeExecutor skyframeExecutor;
  private final boolean keepGoing;
  private final int numJobs;
  private final boolean finalizeActionsToOutputService;
  private final ModifiedFileSet modifiedOutputFiles;
  private final ActionInputFileCache fileCache;
  private final ActionCacheChecker actionCacheChecker;
  private final int progressReportInterval;

  @VisibleForTesting
  public SkyframeBuilder(SkyframeExecutor skyframeExecutor, ActionCacheChecker actionCacheChecker,
      boolean keepGoing, int numJobs, ModifiedFileSet modifiedOutputFiles,
      boolean finalizeActionsToOutputService, ActionInputFileCache fileCache,
      int progressReportInterval) {
    this.skyframeExecutor = skyframeExecutor;
    this.actionCacheChecker = actionCacheChecker;
    this.keepGoing = keepGoing;
    this.numJobs = numJobs;
    this.finalizeActionsToOutputService = finalizeActionsToOutputService;
    this.modifiedOutputFiles = modifiedOutputFiles;
    this.fileCache = fileCache;
    this.progressReportInterval = progressReportInterval;
  }

  @Override
  public void buildArtifacts(
      Reporter reporter,
      Set<Artifact> artifacts,
      Set<ConfiguredTarget> parallelTests,
      Set<ConfiguredTarget> exclusiveTests,
      Collection<ConfiguredTarget> targetsToBuild,
      Collection<AspectValue> aspects,
      Executor executor,
      Set<ConfiguredTarget> builtTargets,
      boolean explain,
      @Nullable Range<Long> lastExecutionTimeRange)
      throws BuildFailedException, AbruptExitException, TestExecException, InterruptedException {
    skyframeExecutor.prepareExecution(modifiedOutputFiles, lastExecutionTimeRange);
    skyframeExecutor.setFileCache(fileCache);
    // Note that executionProgressReceiver accesses builtTargets concurrently (after wrapping in a
    // synchronized collection), so unsynchronized access to this variable is unsafe while it runs.
    ExecutionProgressReceiver executionProgressReceiver =
        new ExecutionProgressReceiver(Preconditions.checkNotNull(builtTargets),
            countTestActions(exclusiveTests), skyframeExecutor.getEventBus());
    ResourceManager.instance().setEventBus(skyframeExecutor.getEventBus());

    boolean success = false;
    EvaluationResult<?> result;

    ActionExecutionStatusReporter statusReporter = ActionExecutionStatusReporter.create(
        reporter, executor, skyframeExecutor.getEventBus());

    AtomicBoolean isBuildingExclusiveArtifacts = new AtomicBoolean(false);
    ActionExecutionInactivityWatchdog watchdog = new ActionExecutionInactivityWatchdog(
        executionProgressReceiver.createInactivityMonitor(statusReporter),
        executionProgressReceiver.createInactivityReporter(statusReporter,
            isBuildingExclusiveArtifacts), progressReportInterval);

    skyframeExecutor.setActionExecutionProgressReportingObjects(executionProgressReceiver,
        executionProgressReceiver, statusReporter);
    watchdog.start();

    try {
      result =
          skyframeExecutor.buildArtifacts(
              reporter,
              executor,
              artifacts,
              targetsToBuild,
              aspects,
              parallelTests,
              /*exclusiveTesting=*/ false,
              keepGoing,
              explain,
              finalizeActionsToOutputService,
              numJobs,
              actionCacheChecker,
              executionProgressReceiver);
      // progressReceiver is finished, so unsynchronized access to builtTargets is now safe.
      success = processResult(reporter, result, keepGoing, skyframeExecutor);

      Preconditions.checkState(
          !success
              || result.keyNames().size()
                  == (artifacts.size()
                      + targetsToBuild.size()
                      + aspects.size()
                      + parallelTests.size()),
          "Build reported as successful but not all artifacts and targets built: %s, %s",
          result,
          artifacts);

      // Run exclusive tests: either tagged as "exclusive" or is run in an invocation with
      // --test_output=streamed.
      isBuildingExclusiveArtifacts.set(true);
      for (ConfiguredTarget exclusiveTest : exclusiveTests) {
        // Since only one artifact is being built at a time, we don't worry about an artifact being
        // built and then the build being interrupted.
        result =
            skyframeExecutor.buildArtifacts(
                reporter,
                executor,
                ImmutableSet.<Artifact>of(),
                targetsToBuild,
                aspects,
                ImmutableSet.of(exclusiveTest), /*exclusiveTesting=*/
                true,
                keepGoing,
                explain,
                finalizeActionsToOutputService,
                numJobs,
                actionCacheChecker,
                null);
        boolean exclusiveSuccess = processResult(reporter, result, keepGoing, skyframeExecutor);
        Preconditions.checkState(!exclusiveSuccess || !result.keyNames().isEmpty(),
            "Build reported as successful but test %s not executed: %s",
            exclusiveTest, result);
        success &= exclusiveSuccess;
      }
    } finally {
      watchdog.stop();
      ResourceManager.instance().unsetEventBus();
      skyframeExecutor.setActionExecutionProgressReportingObjects(null, null, null);
      statusReporter.unregisterFromEventBus();
    }

    if (!success) {
      throw new BuildFailedException();
    }
  }

  /**
   * Process the Skyframe update, taking into account the keepGoing setting.
   *
   * <p>Returns false if the update() failed, but we should continue. Returns true on success.
   * Throws on fail-fast failures.
   */
  private static boolean processResult(EventHandler eventHandler, EvaluationResult<?> result,
      boolean keepGoing, SkyframeExecutor skyframeExecutor)
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
        // keepGoing doesn't throw if there were just ordinary errors.
        return false;
      }
      ErrorInfo errorInfo = Preconditions.checkNotNull(result.getError(), result);
      Exception exception = errorInfo.getException();
      if (exception == null) {
        Preconditions.checkState(!Iterables.isEmpty(errorInfo.getCycleInfo()), errorInfo);
        // If a keepGoing=false build found a cycle, that means there were no other errors thrown
        // during evaluation (otherwise, it wouldn't have bothered to find a cycle). So the best
        // we can do is throw a generic build failure exception, since we've already reported the
        // cycles above.
        throw new BuildFailedException(null, /*hasCatastrophe=*/ false);
      } else {
        rethrow(exception);
      }
    }
    return true;
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
}
