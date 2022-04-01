// Copyright 2022 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.auto.value.AutoValue;
import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.config.BuildConfigurationValue;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.buildeventstream.BuildEventStreamProtos.BuildEventId.ConfigurationId;
import com.google.devtools.build.lib.causes.AnalysisFailedCause;
import com.google.devtools.build.lib.causes.Cause;
import com.google.devtools.build.lib.causes.LabelCause;
import com.google.devtools.build.lib.causes.LoadingFailedCause;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.collect.nestedset.NestedSetBuilder;
import com.google.devtools.build.lib.collect.nestedset.Order;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.packages.NoSuchPackageException;
import com.google.devtools.build.lib.packages.NoSuchTargetException;
import com.google.devtools.build.lib.pkgcache.LoadingFailureEvent;
import com.google.devtools.build.lib.server.FailureDetails.Analysis;
import com.google.devtools.build.lib.server.FailureDetails.Analysis.Code;
import com.google.devtools.build.lib.server.FailureDetails.Execution;
import com.google.devtools.build.lib.server.FailureDetails.FailureDetail;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ConflictException;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.TopLevelAspectsKey;
import com.google.devtools.build.lib.util.DetailedExitCode;
import com.google.devtools.build.lib.util.DetailedExitCode.DetailedExitCodeComparator;
import com.google.devtools.build.skyframe.CycleInfo;
import com.google.devtools.build.skyframe.CyclesReporter;
import com.google.devtools.build.skyframe.ErrorInfo;
import com.google.devtools.build.skyframe.EvaluationResult;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.devtools.build.skyframe.WalkableGraph;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/** A utility class that provides methods to parse errors from Skyframe EvaluationResults. */
public final class SkyframeErrorProcessor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private SkyframeErrorProcessor() {}

  /**
   * Indicates if there are errors with the various phases, and an exception to be thrown to halt
   * the build, in case of --nokeep_going.
   */
  @AutoValue
  abstract static class ErrorProcessingResult {
    abstract boolean hasLoadingError();

    abstract boolean hasAnalysisError();

    abstract ImmutableMap<ActionAnalysisMetadata, ConflictException> actionConflicts();

    @Nullable
    abstract DetailedExitCode executionDetailedExitCode();

    static ErrorProcessingResult create(
        boolean hasLoadingError,
        boolean hasAnalysisError,
        ImmutableMap<ActionAnalysisMetadata, ConflictException> actionConflicts,
        @Nullable DetailedExitCode executionDetailedExitCode) {
      return new AutoValue_SkyframeErrorProcessor_ErrorProcessingResult(
          hasLoadingError, hasAnalysisError, actionConflicts, executionDetailedExitCode);
    }
  }

  /**
   * Process only loading/analysis errors. Returns a {@link ErrorProcessingResult}.
   *
   * <p>In case of --nokeep_going or catastrophe: immediately throw the exception.
   */
  static ErrorProcessingResult processAnalysisErrors(
      EvaluationResult<? extends SkyValue> result,
      Supplier<Map<BuildConfigurationKey, BuildConfigurationValue>> configurationLookupSupplier,
      CyclesReporter cyclesReporter,
      ExtendedEventHandler eventHandler,
      boolean keepGoing,
      @Nullable EventBus eventBus,
      BugReporter bugReporter)
      throws InterruptedException, ViewCreationFailedException {
    try {
      return processErrors(
          result,
          configurationLookupSupplier,
          cyclesReporter,
          eventHandler,
          keepGoing,
          eventBus,
          bugReporter,
          /*includeExecutionPhase=*/ false);
    } catch (BuildFailedException | TestExecException unexpected) {
      throw new IllegalStateException("Unexpected execution phase exception: ", unexpected);
    }
  }

  /**
   * Process errors encountered during analysis/execution, and return a {@link
   * ErrorProcessingResult}.
   *
   * <p>Visible only for use by tests via {@link
   * SkyframeExecutor#getConfiguredTargetMapForTesting(ExtendedEventHandler,
   * BuildConfigurationValue, Iterable)}. When called there, {@code eventBus} must be null to
   * indicate that this is a test, and so there may be additional {@link SkyKey}s in the {@code
   * result} that are not {@link AspectKeyCreator}s or {@link ConfiguredTargetKey}s. Those keys will
   * be ignored.
   *
   * <p>In case of --nokeep_going: immediately throw the exception.
   */
  static ErrorProcessingResult processErrors(
      EvaluationResult<? extends SkyValue> result,
      Supplier<Map<BuildConfigurationKey, BuildConfigurationValue>> configurationLookupSupplier,
      CyclesReporter cyclesReporter,
      ExtendedEventHandler eventHandler,
      boolean keepGoing,
      @Nullable EventBus eventBus,
      @Nullable BugReporter bugReporter,
      boolean includeExecutionPhase)
      throws InterruptedException, ViewCreationFailedException, BuildFailedException,
          TestExecException {
    if (result.getCatastrophe() != null) {
      rethrow(result.getCatastrophe(), bugReporter, result);
    }
    boolean inTest = eventBus == null;
    boolean hasLoadingError = false;
    // At this point, we consider the build to already have an analysis error, unless the error
    // turns out to be with execution.
    boolean hasAnalysisError = true;
    ViewCreationFailedException noKeepGoingAnalysisExceptionAspect = null;
    DetailedExitCode detailedExitCode = null;
    Throwable undetailedCause = null;
    Map<ActionAnalysisMetadata, ConflictException> actionConflicts = new HashMap<>();
    for (Map.Entry<SkyKey, ErrorInfo> errorEntry : result.errorMap().entrySet()) {
      SkyKey errorKey = getErrorKey(errorEntry);
      ErrorInfo errorInfo = errorEntry.getValue();
      if (includeExecutionPhase) {
        assertValidAnalysisOrExecutionException(errorInfo, errorKey, result.getWalkableGraph());
      } else {
        assertValidAnalysisException(errorInfo, errorKey, result.getWalkableGraph());
      }
      cyclesReporter.reportCycles(errorInfo.getCycleInfo(), errorKey, eventHandler);
      Exception cause = errorInfo.getException();
      Preconditions.checkState(cause != null || !errorInfo.getCycleInfo().isEmpty(), errorInfo);

      // TODO(leba): Can the handling of aspect errors be simplified here?
      if (errorKey.argument() instanceof TopLevelAspectsKey) {
        if (includeExecutionPhase && cause instanceof TopLevelConflictException) {
          TopLevelConflictException tlce = (TopLevelConflictException) cause;
          actionConflicts.putAll(tlce.getTransitiveActionConflicts());
        }
        // We skip Aspects in the keepGoing case; the failures should already have been reported to
        // the event handler.
        if (!keepGoing && noKeepGoingAnalysisExceptionAspect == null) {
          TopLevelAspectsKey aspectKey = (TopLevelAspectsKey) errorKey.argument();
          String errorMsg =
              String.format(
                  "Analysis of aspects '%s' failed; build aborted", aspectKey.getDescription());
          if (!(cause instanceof TopLevelConflictException)) {
            noKeepGoingAnalysisExceptionAspect = createViewCreationFailedException(cause, errorMsg);
          }
        }
        continue;
      }

      if (inTest && !(errorKey.argument() instanceof ConfiguredTargetKey)) {
        // This means that we are in a BuildViewTestCase.
        //
        // Tests don't call target pattern parsing before requesting the analysis of a target.
        // Therefore if the package that contains them cannot be loaded, we get an error key that's
        // not a ConfiguredTargetKey, which cannot happen in production code.
        //
        // If it's an existing target in a nonexistent package, the error is signaled by posting an
        // AnalysisFailureEvent on the event bus, which is null in when running a BuildViewTestCase,
        // so we emit the root cause labels directly to the event handler below.
        eventHandler.handle(Event.error(errorInfo.toString()));
        continue;
      }
      Preconditions.checkState(
          errorKey.argument() instanceof ConfiguredTargetKey,
          "expected '%s' to be a TopLevelAspectsKey or ConfiguredTargetKey",
          errorKey.argument());
      ConfiguredTargetKey label = (ConfiguredTargetKey) errorKey.argument();
      Label topLevelLabel = label.getLabel();

      NestedSet<Cause> rootCauses = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      if (cause instanceof ConfiguredValueCreationException) {
        ConfiguredValueCreationException ctCause = (ConfiguredValueCreationException) cause;
        // Previously, the nested set was de-duplicating loading root cause labels. Now that we
        // track Cause instances including a message, we get one event per label and message. In
        // order to keep backwards compatibility, we de-duplicate root cause labels here.
        // TODO(ulfjack): Remove this code once we've migrated to the BEP.
        Set<Label> loadingRootCauses = new HashSet<>();
        for (Cause rootCause : ctCause.getRootCauses().toList()) {
          if (rootCause instanceof LoadingFailedCause) {
            hasLoadingError = true;
            loadingRootCauses.add(rootCause.getLabel());
          }
        }
        if (!inTest) {
          for (Label loadingRootCause : loadingRootCauses) {
            // This event is only for backwards compatibility with the old event protocol. Remove
            // once we've migrated to the build event protocol.
            eventBus.post(new LoadingFailureEvent(topLevelLabel, loadingRootCause));
          }
        }
        rootCauses = ctCause.getRootCauses();
      } else if (!errorInfo.getCycleInfo().isEmpty()) {
        Label analysisRootCause =
            maybeGetConfiguredTargetCycleCulprit(topLevelLabel, errorInfo.getCycleInfo());
        rootCauses =
            analysisRootCause != null
                ? NestedSetBuilder.create(
                    Order.STABLE_ORDER,
                    new LabelCause(
                        analysisRootCause,
                        DetailedExitCode.of(createFailureDetail("Dependency cycle", Code.CYCLE))))
                // TODO(ulfjack): We need to report the dependency cycle here. How?
                : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      } else if (cause instanceof ActionConflictException) {
        ((ActionConflictException) cause).reportTo(eventHandler);
        rootCauses = NestedSetBuilder.emptySet(Order.STABLE_ORDER);
      } else if (cause instanceof NoSuchPackageException) {
        // This branch is only taken in --nokeep_going builds. In a --keep_going build, the
        // AnalysisFailedCause is properly reported through the ConfiguredValueCreationException.
        BuildConfigurationValue configuration =
            configurationLookupSupplier.get().get(label.getConfigurationKey());
        ConfigurationId configId = configuration.getEventId().getConfiguration();
        AnalysisFailedCause analysisFailedCause =
            new AnalysisFailedCause(
                topLevelLabel, configId, ((NoSuchPackageException) cause).getDetailedExitCode());
        rootCauses = NestedSetBuilder.create(Order.STABLE_ORDER, analysisFailedCause);
      } else if (cause instanceof TopLevelConflictException) {
        // The root causes for action conflicts will be determined and reported later in the error
        // handling process.
        TopLevelConflictException tlce = (TopLevelConflictException) cause;
        actionConflicts.putAll(tlce.getTransitiveActionConflicts());
        continue;
      } else if (isExecutionException(cause)) {
        detailedExitCode =
            DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
                detailedExitCode, ((DetailedException) cause).getDetailedExitCode());
        rootCauses =
            cause instanceof ActionExecutionException
                ? ((ActionExecutionException) cause).getRootCauses()
                : NestedSetBuilder.emptySet(Order.STABLE_ORDER);
        hasAnalysisError = false;
      } else {
        // TODO(ulfjack): Report something!
        undetailedCause = cause;
      }

      if (!inTest) {
        BuildConfigurationValue configuration =
            configurationLookupSupplier.get().get(label.getConfigurationKey());
        eventBus.post(
            new AnalysisFailureEvent(
                label, configuration == null ? null : configuration.getEventId(), rootCauses));
      } else {
        // eventBus is null, but test can still assert on the expected root causes being found.
        eventHandler.handle(Event.error(rootCauses.toList().toString()));
      }

      if (!keepGoing) {
        if (isExecutionException(cause)) {
          rethrow(cause, bugReporter, result);
        }
        String errorMsg =
            includeExecutionPhase
                ? String.format("Build of target '%s' failed; build aborted", topLevelLabel)
                : String.format("Analysis of target '%s' failed; build aborted", topLevelLabel);
        throw createViewCreationFailedException(cause, errorMsg);
      } else {
        String warningMsg =
            includeExecutionPhase
                ? String.format("errors encountered while building target '%s'", topLevelLabel)
                : String.format(
                    "errors encountered while analyzing target '%s': it will not be built",
                    topLevelLabel);
        eventHandler.handle(Event.warn(warningMsg));
      }
    }

    // We wait until this point to throw the exception to make sure that the target error is
    // preferred over the aspect one, for the same target.
    if (!keepGoing && noKeepGoingAnalysisExceptionAspect != null) {
      throw noKeepGoingAnalysisExceptionAspect;
    }

    if (includeExecutionPhase && detailedExitCode == null) {
      detailedExitCode = createDetailedExitCodeForUndetailedExecutionCause(result, undetailedCause);
    }

    return ErrorProcessingResult.create(
        hasLoadingError, hasAnalysisError, ImmutableMap.copyOf(actionConflicts), detailedExitCode);
  }

  private static SkyKey getErrorKey(Entry<SkyKey, ErrorInfo> errorEntry) {
    if (errorEntry.getKey().argument() instanceof BuildDriverKey) {
      return ((BuildDriverKey) errorEntry.getKey().argument()).getActionLookupKey();
    }
    return errorEntry.getKey();
  }

  private static ViewCreationFailedException createViewCreationFailedException(
      @Nullable Exception e, String errorMsg) {
    if (e == null) {
      return new ViewCreationFailedException(
          errorMsg, createFailureDetail(errorMsg + " due to cycle", Code.CYCLE));
    }
    return new ViewCreationFailedException(
        errorMsg, maybeContextualizeFailureDetail(e, errorMsg), e);
  }

  /**
   * Returns a {@link FailureDetail} with message prefixed by {@code errorMsg} derived from the
   * failure detail in {@code e} if it's a {@link DetailedException}, and otherwise returns one with
   * {@code errorMsg} and {@link Code#UNEXPECTED_ANALYSIS_EXCEPTION}.
   */
  private static FailureDetail maybeContextualizeFailureDetail(
      @Nullable Exception e, String errorMsg) {
    DetailedException detailedException = convertToAnalysisException(e);
    if (detailedException == null) {
      return createFailureDetail(errorMsg, Code.UNEXPECTED_ANALYSIS_EXCEPTION);
    }
    FailureDetail originalFailureDetail =
        detailedException.getDetailedExitCode().getFailureDetail();
    return originalFailureDetail.toBuilder()
        .setMessage(errorMsg + ": " + originalFailureDetail.getMessage())
        .build();
  }

  private static FailureDetail createFailureDetail(String errorMessage, Code code) {
    return FailureDetail.newBuilder()
        .setMessage(errorMessage)
        .setAnalysis(Analysis.newBuilder().setCode(code))
        .build();
  }

  @Nullable
  private static Label maybeGetConfiguredTargetCycleCulprit(
      Label labelToLoad, Iterable<CycleInfo> cycleInfos) {
    for (CycleInfo cycleInfo : cycleInfos) {
      SkyKey culprit = Iterables.getFirst(cycleInfo.getCycle(), null);
      if (culprit == null) {
        continue;
      }
      if (culprit.functionName().equals(SkyFunctions.CONFIGURED_TARGET)) {
        return ((ConfiguredTargetKey) culprit.argument()).getLabel();
      } else if (culprit.functionName().equals(TransitiveTargetKey.NAME)) {
        return ((TransitiveTargetKey) culprit).getLabel();
      } else {
        return labelToLoad;
      }
    }
    return null;
  }

  private static void assertValidAnalysisException(
      ErrorInfo errorInfo, SkyKey key, WalkableGraph walkableGraph) throws InterruptedException {
    Throwable cause = errorInfo.getException();
    if (cause == null) {
      // Cycle.
      return;
    }

    if (convertToAnalysisException(cause) != null) {
      // Valid exception type.
      return;
    }

    logUnexpectedExceptionOrigin(errorInfo, key, walkableGraph, cause);
  }

  private static void assertValidAnalysisOrExecutionException(
      ErrorInfo errorInfo, SkyKey key, WalkableGraph walkableGraph) throws InterruptedException {
    Throwable cause = errorInfo.getException();
    if (cause == null) {
      // Cycle.
      return;
    }

    if (convertToAnalysisException(cause) != null
        || isExecutionException(cause)
        || cause instanceof TopLevelConflictException) {
      // Valid exception type.
      return;
    }

    logUnexpectedExceptionOrigin(errorInfo, key, walkableGraph, cause);
  }

  /**
   * Walk the graph to find a path to the lowest-level node that threw unexpected exception and log
   * it.
   */
  private static void logUnexpectedExceptionOrigin(
      ErrorInfo errorInfo, SkyKey key, WalkableGraph walkableGraph, Throwable cause)
      throws InterruptedException {
    List<SkyKey> path = new ArrayList<>();
    try {
      SkyKey currentKey = key;
      boolean foundDep;
      do {
        path.add(currentKey);
        foundDep = false;

        Map<SkyKey, Exception> missingMap =
            walkableGraph.getMissingAndExceptions(ImmutableList.of(currentKey));
        if (missingMap.containsKey(currentKey) && missingMap.get(currentKey) == null) {
          // This can happen in a no-keep-going build, where we don't write the bubbled-up error
          // nodes to the graph.
          break;
        }

        for (SkyKey dep : walkableGraph.getDirectDeps(currentKey)) {
          if (cause.equals(walkableGraph.getException(dep))) {
            currentKey = dep;
            foundDep = true;
            break;
          }
        }
      } while (foundDep);
    } finally {
      BugReport.logUnexpected("Unexpected analysis error: %s -> %s, (%s)", key, errorInfo, path);
    }
  }

  @Nullable
  private static DetailedException convertToAnalysisException(Throwable cause) {
    // The cause may be NoSuch{Target,Package}Exception if we run the reduced loading phase and then
    // analyze with --nokeep_going.
    if (cause instanceof SaneAnalysisException
        || cause instanceof NoSuchTargetException
        || cause instanceof NoSuchPackageException) {
      return (DetailedException) cause;
    }
    return null;
  }

  private static boolean isExecutionException(Throwable cause) {
    return cause instanceof ActionExecutionException
        || cause instanceof InputFileErrorException
        || cause instanceof TestExecException;
  }

  /**
   * Process an {@link EvaluationResult}, taking into account the keepGoing setting.
   *
   * <p>Returns a nullable {@link DetailedExitCode} value, as follows:
   *
   * <ol>
   *   <li>{@code null}, if {@code result} had no errors
   *   <li>{@code e} if result had errors and one of them specified a {@link DetailedExitCode} value
   *       {@code e}
   *   <li>a {@link DetailedExitCode} with {@link Execution.Code#NON_ACTION_EXECUTION_FAILURE} if
   *       result had errors but none specified a {@link DetailedExitCode} value
   * </ol>
   *
   * <p>Throws on catastrophic failures and, if !keepGoing, on any failure. TODO(leba): We should
   * ideally remove this method and incorporate its logic into #processAnalysisErrors.
   */
  @Nullable
  public static DetailedExitCode processResult(
      ExtendedEventHandler eventHandler,
      EvaluationResult<?> result,
      boolean keepGoing,
      CyclesReporter cyclesReporter,
      @Nullable BugReporter bugReporter)
      throws BuildFailedException, TestExecException {
    if (result.hasError()) {
      for (Map.Entry<SkyKey, ErrorInfo> entry : result.errorMap().entrySet()) {
        ImmutableList<CycleInfo> cycles = entry.getValue().getCycleInfo();
        cyclesReporter.reportCycles(cycles, entry.getKey(), eventHandler);
      }

      if (result.getCatastrophe() != null) {
        rethrow(result.getCatastrophe(), bugReporter, result);
      }
      if (keepGoing) {
        return getDetailedExitCode(result);
      }
      ErrorInfo errorInfo = Preconditions.checkNotNull(result.getError(), result);
      Exception exception = errorInfo.getException();
      if (exception == null) {
        Preconditions.checkState(!errorInfo.getCycleInfo().isEmpty(), errorInfo);
        // If a keepGoing=false build found a cycle, that means there were no other errors thrown
        // during evaluation (otherwise, it wouldn't have bothered to find a cycle). So the best
        // we can do is throw a generic build failure exception, since we've already reported the
        // cycles above.
        throw new BuildFailedException(null, CYCLE_CODE);
      } else {
        rethrow(exception, bugReporter, result);
      }
    }

    return null;
  }

  private static DetailedExitCode getDetailedExitCode(EvaluationResult<?> result) {
    // If build fails and keepGoing is true, an exit code is assigned using reported errors
    // in the following order:
    //   1. First infrastructure error with non-null exit code
    //   2. First non-infrastructure error with non-null exit code
    //   3. If the build fails but no interpretable error is specified, BUILD_FAILURE.
    DetailedExitCode detailedExitCode = null;
    Throwable undetailedCause = null;
    for (Map.Entry<SkyKey, ErrorInfo> error : result.errorMap().entrySet()) {
      Throwable cause = error.getValue().getException();
      if (cause instanceof DetailedException) {
        // Update global exit code when current exit code is not null and global exit code has
        // a lower 'reporting' priority.
        detailedExitCode =
            DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
                detailedExitCode, ((DetailedException) cause).getDetailedExitCode());
        if (!(cause instanceof ActionExecutionException)
            && !(cause instanceof InputFileErrorException)) {
          logger.atWarning().withCause(cause).log(
              "Non-action-execution/input-error exception for %s", error);
        }
      } else if (!error.getValue().getCycleInfo().isEmpty()) {
        detailedExitCode =
            DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
                detailedExitCode, CYCLE_CODE);
      } else {
        undetailedCause = cause;
      }
    }
    if (detailedExitCode != null) {
      return detailedExitCode;
    }
    return createDetailedExitCodeForUndetailedExecutionCause(result, undetailedCause);
  }

  /**
   * Figure out why an action's analysis/execution failed and rethrow the right kind of exception.
   * At the moment, we don't expect any analysis failure to be catastrophic.
   */
  @VisibleForTesting
  static void rethrow(
      Throwable cause, BugReporter bugReporter, EvaluationResult<?> resultForDebugging)
      throws BuildFailedException, TestExecException {
    Throwables.throwIfUnchecked(cause);
    Throwable innerCause = cause.getCause();
    if (innerCause instanceof TestExecException) {
      throw (TestExecException) innerCause;
    }
    if (cause instanceof ActionExecutionException) {
      ActionExecutionException actionExecutionCause = (ActionExecutionException) cause;
      String message = cause.getMessage();
      if (actionExecutionCause.getAction() != null) {
        message = actionExecutionCause.getAction().describe() + " failed: " + message;
      }
      // Sometimes ActionExecutionExceptions are caused by Actions with no owner.
      if (actionExecutionCause.getLocation() != null) {
        message = actionExecutionCause.getLocation() + " " + message;
      }
      throw new BuildFailedException(
          message,
          actionExecutionCause.isCatastrophe(),
          /*errorAlreadyShown=*/ !actionExecutionCause.showError(),
          actionExecutionCause.getDetailedExitCode());
    }
    if (cause instanceof InputFileErrorException) {
      throw (InputFileErrorException) cause;
    }

    // We encountered an exception we don't think we should have encountered. This can indicate
    // an exception-processing bug in our code, such as lower level exceptions not being properly
    // handled, or in our expectations in this method.

    if (cause instanceof DetailedException) {
      // The exception escaped Skyframe error bubbling, but its failure detail can still be used.
      bugReporter.logUnexpected(
          (Exception) cause,
          "action terminated with unexpected exception with result %s",
          resultForDebugging);
      throw new BuildFailedException(
          cause.getMessage(), ((DetailedException) cause).getDetailedExitCode());
    }

    // An undetailed exception means we may incorrectly attribute responsibility for the failure:
    // we need to fix that.
    bugReporter.sendBugReport(
        new IllegalStateException(
            "action terminated with unexpected exception with result " + resultForDebugging,
            cause));
    String message =
        "Unexpected exception, please file an issue with the Bazel team: " + cause.getMessage();
    throw new BuildFailedException(
        message, createDetailedExecutionExitCode(message, UNKONW_EXECUTION));
  }

  private static DetailedExitCode createDetailedExitCodeForUndetailedExecutionCause(
      EvaluationResult<?> result, Throwable undetailedCause) {
    // TODO(b/227660368): These warning logs should be a bug report, but tests currently fail.
    if (undetailedCause == null) {
      logger.atWarning().log("No exceptions found despite error in %s", result);
      return createDetailedExecutionExitCode(
          "keep_going execution failed without an action failure",
          Execution.Code.NON_ACTION_EXECUTION_FAILURE);
    }
    logger.atWarning().withCause(undetailedCause).log("No detailed exception found in %s", result);
    return createDetailedExecutionExitCode(
        "keep_going execution failed without an action failure: "
            + undetailedCause.getMessage()
            + " ("
            + undetailedCause.getClass().getSimpleName()
            + ")",
        Execution.Code.NON_ACTION_EXECUTION_FAILURE);
  }

  private static final DetailedExitCode CYCLE_CODE =
      createDetailedExecutionExitCode("cycle found during execution", Execution.Code.CYCLE);
  private static final Execution UNKONW_EXECUTION =
      Execution.newBuilder().setCode(Execution.Code.UNEXPECTED_EXCEPTION).build();

  private static DetailedExitCode createDetailedExecutionExitCode(
      String message, Execution.Code detailedCode) {
    return createDetailedExecutionExitCode(
        message, Execution.newBuilder().setCode(detailedCode).build());
  }

  private static DetailedExitCode createDetailedExecutionExitCode(
      String message, Execution execution) {
    return DetailedExitCode.of(
        FailureDetail.newBuilder().setMessage(message).setExecution(execution).build());
  }
}
