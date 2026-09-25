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

import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.devtools.build.lib.buildeventstream.BuildEventIdUtil.configurationIdMessage;
import static com.google.devtools.build.lib.skyframe.ActionArtifactCycleReporter.ACTION_OR_ARTIFACT_OR_TRANSITIVE_RDEP;
import static java.util.Objects.requireNonNull;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.eventbus.EventBus;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionConflictException;
import com.google.devtools.build.lib.actions.ActionExecutionException;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.BuildFailedException;
import com.google.devtools.build.lib.actions.InputFileErrorException;
import com.google.devtools.build.lib.actions.TestExecException;
import com.google.devtools.build.lib.actions.TopLevelOutputException;
import com.google.devtools.build.lib.analysis.AnalysisFailureEvent;
import com.google.devtools.build.lib.analysis.ViewCreationFailedException;
import com.google.devtools.build.lib.analysis.constraints.TopLevelConstraintSemantics.TargetCompatibilityCheckException;
import com.google.devtools.build.lib.bazel.bzlmod.ExternalDepsException;
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.bugreport.BugReporter;
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
import com.google.devtools.build.lib.skyframe.ArtifactNestedSetFunction.ArtifactNestedSetEvalException;
import com.google.devtools.build.lib.skyframe.AspectCompletionValue.AspectCompletionKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectBaseKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectKey;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.TopLevelAspectsKey;
import com.google.devtools.build.lib.skyframe.TargetCompletionValue.TargetCompletionKey;
import com.google.devtools.build.lib.skyframe.TestCompletionValue.TestCompletionKey;
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
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/** A utility class that provides methods to parse errors from Skyframe EvaluationResults. */
public final class SkyframeErrorProcessor {

  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private SkyframeErrorProcessor() {}

  /**
   * Indicates if there are errors with the various phases, and an exception to be thrown to halt
   * the build, in case of --nokeep_going.
   *
   * <p>The various attributes will be used later on to construct the FailureDetail in {@link
   * com.google.devtools.build.lib.analysis.BuildView#createAnalysisFailureDetail}.
   *
   * @param hasLoadingError whether there are loading errors.
   * @param hasAnalysisError whether there are any non-execution errors. Note that this includes
   *     loading errors and action conflicts.
   * @param actionConflicts the action conflicts encountered during analysis.
   * @param executionDetailedExitCode the detailed exit code for execution errors. This is
   *     <ul>
   *       <li>{@code null}, if {@code result} had no errors or the errors were all analysis errors.
   *       <li>the most important {@link DetailedExitCode} among the execution errors that specified
   *           one, ranked by {@link DetailedExitCodeComparator}
   *       <li>a {@link DetailedExitCode} with {@link Execution.Code#UNEXPECTED_EXCEPTION} if an
   *           execution error specified no {@link DetailedExitCode} at all
   *     </ul>
   *
   * @param aspectKeysForConflictReporting the aspect keys for conflict reporting.
   */
  public record ErrorProcessingResult(
      boolean hasLoadingError,
      boolean hasAnalysisError,
      ImmutableMap<ActionAnalysisMetadata, ActionConflictException> actionConflicts,
      @Nullable DetailedExitCode executionDetailedExitCode,
      ImmutableList<ActionLookupKey> aspectKeysForConflictReporting) {
    public ErrorProcessingResult {
      requireNonNull(actionConflicts, "actionConflicts");
      requireNonNull(aspectKeysForConflictReporting, "aspectKeysForConflictReporting");
    }

    static AggregatingBuilder newBuilder() {
      return new AggregatingBuilder();
    }

    static class AggregatingBuilder {
      private boolean hasLoadingError = false;
      private boolean hasAnalysisError = false;
      private ImmutableMap<ActionAnalysisMetadata, ActionConflictException> actionConflicts =
          ImmutableMap.of();
      @Nullable private DetailedExitCode executionDetailedExitCode = null;
      private ImmutableList<ActionLookupKey> aspectKeysForConflictReporting = ImmutableList.of();

      private void aggregateSingleResult(ClassifiedError error) {
        hasLoadingError = hasLoadingError || error.isLoadingError();
        hasAnalysisError = hasAnalysisError || error.isAnalysisError();
        executionDetailedExitCode =
            DetailedExitCodeComparator.chooseMoreImportantWithFirstIfTie(
                executionDetailedExitCode, error.executionDetailedExitCode());
      }

      private void addConflicts(ConflictHarvest harvest) {
        actionConflicts = harvest.actionConflicts();
        aspectKeysForConflictReporting = harvest.aspectKeysForConflictReporting();
        // An action conflict is an analysis error, even though SkyframeBuildView is the one that
        // reports it.
        hasAnalysisError = hasAnalysisError || !harvest.actionConflicts().isEmpty();
      }

      ErrorProcessingResult build() {
        return new ErrorProcessingResult(
            hasLoadingError,
            hasAnalysisError,
            actionConflicts,
            executionDetailedExitCode,
            aspectKeysForConflictReporting);
      }
    }
  }

  /**
   * Which error wins when {@code --nokeep_going} has to pick a single one to abort the build with.
   * Lower ordinal wins.
   *
   * <p>This is a reporting order, not a statement about how bad an error is.
   */
  private enum ReportingPriority {
    EXECUTION,
    TARGET_ANALYSIS,
    ASPECT_ANALYSIS
  }

  /**
   * One error from {@link EvaluationResult#errorMap}, normalized: the wrapper key peeled off, the
   * exception classified and the root causes extracted. These are the building blocks of the final
   * {@link ErrorProcessingResult}.
   *
   * <p>Classifying an error makes no decisions, posts nothing and throws nothing.
   *
   * @param normalizedKey the error key with any wrapper peeled off by {@link
   *     #getEffectiveErrorKey}, not the key Skyframe reported the error under.
   * @param cause the exception that failed the key, or null if it failed because of a cycle.
   * @param executionDetailedExitCode non-null iff {@code reportingPriority} is {@link
   *     ReportingPriority#EXECUTION}.
   * @param loadingRootCauses the labels of the {@link LoadingFailedCause}s among {@code
   *     analysisRootCauses}, de-duplicated.
   */
  private record ClassifiedError(
      SkyKey normalizedKey,
      @Nullable Exception cause,
      ReportingPriority reportingPriority,
      @Nullable DetailedExitCode executionDetailedExitCode,
      NestedSet<Cause> analysisRootCauses,
      ImmutableSet<Label> loadingRootCauses) {

    static ClassifiedError execution(
        SkyKey normalizedKey,
        @Nullable Exception cause,
        DetailedExitCode executionDetailedExitCode,
        NestedSet<Cause> analysisRootCauses) {
      return new ClassifiedError(
          normalizedKey,
          cause,
          ReportingPriority.EXECUTION,
          executionDetailedExitCode,
          analysisRootCauses,
          /* loadingRootCauses= */ ImmutableSet.of());
    }

    static ClassifiedError analysis(
        SkyKey normalizedKey, @Nullable Exception cause, NestedSet<Cause> analysisRootCauses) {
      NestedSet<Cause> rootCauses =
          normalizedKey instanceof AspectBaseKey
              // Aspect errors discard their root causes, so a TopLevelAspectsKey posts an
              // AnalysisFailureEvent with an empty cause set and a bare AspectKey posts nothing at
              // all, even though e.g. AspectCreationException#getCauses would supply real ones.
              // TODO(b/561978611): Stop doing this and treat aspect keys like any other key.
              ? NestedSetBuilder.<Cause>emptySet(Order.STABLE_ORDER)
              : analysisRootCauses;
      return new ClassifiedError(
          normalizedKey,
          cause,
          normalizedKey instanceof AspectBaseKey
              ? ReportingPriority.ASPECT_ANALYSIS
              : ReportingPriority.TARGET_ANALYSIS,
          /* executionDetailedExitCode= */ null,
          rootCauses,
          // A Cause carries a message as well as a label, so the same label can appear more
          // than once. The pre-BEP LoadingFailureEvent protocol only knows about labels, so
          // de-duplicate. TODO(ulfjack): Remove this once we've migrated to the BEP.
          rootCauses.toList().stream()
              .filter(LoadingFailedCause.class::isInstance)
              .map(Cause::getLabel)
              .collect(toImmutableSet()));
    }

    /** A cycle is the only way for a key to fail without an exception. */
    boolean isCycle() {
      return cause == null;
    }

    /** True for all non-execution errors, including loading errors. */
    boolean isAnalysisError() {
      return reportingPriority != ReportingPriority.EXECUTION;
    }

    boolean isLoadingError() {
      return !loadingRootCauses.isEmpty();
    }

    @Nullable
    Label label() {
      return normalizedKey instanceof ActionLookupKey actionLookupKey
          ? actionLookupKey.getLabel()
          : null;
    }
  }

  /**
   * An error that is not an action conflict.
   *
   * @param errorKey the {@linkplain #getEffectiveErrorKey effective} error key, not the key
   *     Skyframe reported the error under
   */
  private record RemainingError(SkyKey errorKey, ErrorInfo errorInfo) {}

  /**
   * The action conflicts found in an {@link EvaluationResult}, and the errors that are left.
   *
   * <p>Action conflicts are reported by {@link SkyframeBuildView} rather than here: at this point
   * the conflict set is still incomplete, and attributing a conflict to a top-level target needs a
   * further Skyframe evaluation.
   *
   * @param remainingErrors the errors that were not conflicts, in the order Skyframe surfaced them.
   *     Everything downstream works from these, and so never has to know that conflicts exist.
   */
  private record ConflictHarvest(
      ImmutableMap<ActionAnalysisMetadata, ActionConflictException> actionConflicts,
      ImmutableList<ActionLookupKey> aspectKeysForConflictReporting,
      ImmutableList<RemainingError> remainingErrors) {}

  private static ConflictHarvest harvestActionConflicts(Map<SkyKey, ErrorInfo> errors) {
    ImmutableMap.Builder<ActionAnalysisMetadata, ActionConflictException> actionConflicts =
        ImmutableMap.builder();
    ImmutableList.Builder<ActionLookupKey> aspectKeys = ImmutableList.builder();
    ImmutableList.Builder<RemainingError> remainingErrors = ImmutableList.builder();

    for (Map.Entry<SkyKey, ErrorInfo> errorEntry : errors.entrySet()) {
      SkyKey errorKey = getEffectiveErrorKey(errorEntry.getKey());
      Exception exception = errorEntry.getValue().getException();
      // A conflict only ever arrives on a top-level target or aspect key.
      if (isValidErrorKeyType(errorKey)) {
        if (exception instanceof TopLevelConflictException tlce) {
          actionConflicts.putAll(tlce.getTransitiveActionConflicts());
          continue;
        }
        if (exception instanceof ActionConflictException ace) {
          actionConflicts.put(ace.getAttemptedAction(), ace);
          if (errorKey instanceof AspectBaseKey && ace.getAspectKey() != null) {
            // SkyframeBuildView reports a conflict against an AspectKey, which it normally derives
            // from the TopLevelAspectsValue. An intra-aspect conflict leaves that value null, so
            // the key has to be carried out of the exception here.
            aspectKeys.add(ace.getAspectKey());
          }
          continue;
        }
      }
      remainingErrors.add(new RemainingError(errorKey, errorEntry.getValue()));
    }

    return new ConflictHarvest(
        // Two top-level targets that share a conflicting dependency each report that dependency's
        // action, so the same key can arrive twice and buildOrThrow() would reject it. Keep the
        // last, which is what accumulating into a map did.
        actionConflicts.buildKeepingLast(), aspectKeys.build(), remainingErrors.build());
  }

  /**
   * Process only loading/analysis errors. Returns a {@link ErrorProcessingResult}.
   *
   * <p>In case of --nokeep_going: immediately throw the exception.
   */
  static ErrorProcessingResult processAnalysisErrors(
      EvaluationResult<? extends SkyValue> result,
      CyclesReporter cyclesReporter,
      ExtendedEventHandler eventHandler,
      boolean keepGoing,
      boolean keepEdges,
      @Nullable EventBus eventBus,
      BugReporter bugReporter)
      throws InterruptedException, ViewCreationFailedException {
    try {
      return processErrors(
          result,
          cyclesReporter,
          eventHandler,
          keepGoing,
          keepEdges,
          eventBus,
          bugReporter,
          /* includeExecutionPhase= */ false);
    } catch (BuildFailedException | TestExecException unexpected) {
      throw new IllegalStateException("Unexpected execution phase exception: ", unexpected);
    }
  }

  /** Process only execution errors. Returns a {@link ErrorProcessingResult}. */
  public static ErrorProcessingResult processExecutionErrors(
      EvaluationResult<? extends SkyValue> result,
      CyclesReporter cyclesReporter,
      ExtendedEventHandler eventHandler,
      boolean keepGoing,
      boolean keepEdges,
      @Nullable EventBus eventBus,
      BugReporter bugReporter)
      throws InterruptedException, BuildFailedException, TestExecException {
    try {
      return processErrors(
          result,
          cyclesReporter,
          eventHandler,
          keepGoing,
          keepEdges,
          eventBus,
          bugReporter,
          /* includeExecutionPhase= */ true);
    } catch (ViewCreationFailedException unexpected) {
      throw new IllegalStateException("Unexpected analysis phase exception: ", unexpected);
    }
  }

  /**
   * Process errors encountered during analysis/execution.
   *
   * <p>Runs in three phases: classify every error, report all of them, then decide what to throw.
   * With --keep_going there is nothing to decide and an {@link ErrorProcessingResult} is returned
   * instead. Action conflicts take none of these phases: reporting one needs information that isn't
   * available yet, so they are handed to {@link SkyframeBuildView} untouched.
   *
   * <p>A null {@code eventBus} indicates that this is a {@code BuildViewTestCase}. Such tests don't
   * parse target patterns before requesting analysis, so the {@code result} may contain {@link
   * SkyKey}s that are neither {@link AspectBaseKey}s nor {@link ConfiguredTargetKey}s, which cannot
   * happen in production. Those keys are reported to the event handler and otherwise ignored.
   *
   * @throws ViewCreationFailedException when the root cause is analysis-related.
   * @throws BuildFailedException when the root cause is execution-related.
   * @throws TestExecException when the root cause is test-related.
   * @return an ErrorProcessingResult (only in --keep_going mode, or action conflict).
   */
  static ErrorProcessingResult processErrors(
      EvaluationResult<? extends SkyValue> result,
      CyclesReporter cyclesReporter,
      ExtendedEventHandler eventHandler,
      boolean keepGoing,
      boolean keepEdges,
      @Nullable EventBus eventBus,
      @Nullable BugReporter bugReporter,
      boolean includeExecutionPhase)
      throws InterruptedException,
          ViewCreationFailedException,
          BuildFailedException,
          TestExecException {
    boolean inBuildViewTest = eventBus == null;

    Map<SkyKey, ErrorInfo> allErrors = result.errorMap();

    // Phase 0: report the cycles and take the action conflicts out.
    //
    // Cycles are reported for every error, including the conflicts harvested below:
    // ErrorInfo#fromChildErrors keeps one child's exception and the cycles of all of them, so an
    // error can be both at once. CyclesReporter deduplicates against the cycles it has already
    // reported, so it has to see them in the order Skyframe surfaced them.
    //
    // The cycle reporter requires that the path to the cycle starts at the top level key
    // (requested via SkyframeExecutor), hence we need to provide the original top level key here.
    //
    // Why is there a need for "original" vs "effective" error key?
    // 1) The non-skymeld code path deals with ActionLookupKeys as the top level key,
    // 2) We wanted to share the error handling code between skymeld and non skymeld.
    // To do so, we need to "normalize" the top level key in Skymeld mode by getting the effective
    // ActionLookupKey from a BuildDriverKey. The rest of the method can then be easily shared.
    for (Map.Entry<SkyKey, ErrorInfo> errorEntry : allErrors.entrySet()) {
      cyclesReporter.reportCycles(
          errorEntry.getValue().getCycleInfo(),
          /* topLevelKey= */ errorEntry.getKey(),
          eventHandler);
    }

    // Action conflicts are pulled out and handed to SkyframeBuildView untouched: reporting them
    // needs information that isn't available yet. Nothing below this point knows they exist.
    ConflictHarvest conflictHarvest = harvestActionConflicts(allErrors);

    // Phase 1: classify. Decides nothing and posts nothing.
    ImmutableList<ClassifiedError> errors =
        classifyAll(
            result,
            conflictHarvest.remainingErrors(),
            eventHandler,
            keepEdges,
            inBuildViewTest,
            bugReporter,
            includeExecutionPhase);

    // Under --nokeep_going, the build aborts with exactly one error, and only that error is
    // reported.
    @Nullable
    ClassifiedError abortWith =
        keepGoing || errors.isEmpty() ? null : Collections.min(errors, NO_KEEP_GOING_PRECEDENCE);

    // Phase 2: report.
    for (ClassifiedError error : abortWith == null ? errors : ImmutableList.of(abortWith)) {
      maybePostFailureEvents(eventHandler, eventBus, inBuildViewTest, error);
      if (keepGoing) {
        logOrPrintWarningsKeepGoing(error, eventHandler);
      }
    }

    // Phase 3: decide.
    if (abortWith != null) {
      abortBuild(result, bugReporter, abortWith);
    }

    ErrorProcessingResult.AggregatingBuilder aggregatingResultBuilder =
        ErrorProcessingResult.newBuilder();
    aggregatingResultBuilder.addConflicts(conflictHarvest);
    errors.forEach(aggregatingResultBuilder::aggregateSingleResult);
    return aggregatingResultBuilder.build();
  }

  /**
   * Validates the non-conflict {@link RemainingError}s from {@link #harvestActionConflicts} and
   * converts each one via {@link #classify} into a {@link ClassifiedError} (recording its {@link
   * ReportingPriority}, {@link DetailedExitCode}, and root causes).
   *
   * <p>In {@code BuildViewTestCase} ({@code inBuildViewTest}), entries whose key is not a valid
   * top-level error key are reported directly to {@code eventHandler} and omitted from the returned
   * list.
   */
  private static ImmutableList<ClassifiedError> classifyAll(
      EvaluationResult<? extends SkyValue> result,
      ImmutableList<RemainingError> remainingErrors,
      ExtendedEventHandler eventHandler,
      boolean keepEdges,
      boolean inBuildViewTest,
      @Nullable BugReporter bugReporter,
      boolean includeExecutionPhase)
      throws InterruptedException {
    ImmutableList.Builder<ClassifiedError> errors =
        ImmutableList.builderWithExpectedSize(remainingErrors.size());
    for (RemainingError remainingError : remainingErrors) {
      SkyKey errorKey = remainingError.errorKey();
      ErrorInfo errorInfo = remainingError.errorInfo();

      if (includeExecutionPhase) {
        assertValidAnalysisOrExecutionException(
            errorInfo, errorKey, result.getWalkableGraph(), keepEdges);
      } else {
        assertValidAnalysisException(errorInfo, errorKey, result.getWalkableGraph(), keepEdges);
      }
      Preconditions.checkState(
          errorInfo.getException() != null || !errorInfo.getCycleInfo().isEmpty(), errorInfo);

      // TODO(b/561978611): Can we remove this divergence?
      if (inBuildViewTest && !isValidErrorKeyType(errorKey)) {
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

      errors.add(classify(result, bugReporter, errorKey, errorInfo));
    }
    return errors.build();
  }

  /**
   * The order in which {@code --nokeep_going} prefers to abort the build. Total, and independent of
   * the order Skyframe surfaced the errors in, so that the same build always fails the same way.
   *
   * <p>The {@link ReportingPriority} ordering ({@code EXECUTION > TARGET_ANALYSIS >
   * ASPECT_ANALYSIS}) preserves legacy precedence. Within {@link ReportingPriority#EXECUTION}, the
   * more important exit code wins ({@code executionDetailedExitCode} is null for analysis errors).
   * Remaining ties are broken by label, then by {@code normalizedKey().toString()} (which tells
   * apart e.g. the same target in different configurations).
   */
  private static final Comparator<ClassifiedError> NO_KEEP_GOING_PRECEDENCE =
      Comparator.comparing(ClassifiedError::reportingPriority)
          .thenComparing(
              ClassifiedError::executionDetailedExitCode,
              DetailedExitCodeComparator.INSTANCE.reversed())
          .thenComparing(error -> error.label() == null ? "" : error.label().toString())
          .thenComparing(error -> error.normalizedKey().toString());

  /**
   * Posts the failure events for a single error.
   *
   * <p>A {@link TopLevelAspectsKey} does get an {@link AnalysisFailureEvent}, attributed to its
   * base configured target, but with no root causes (see {@link ClassifiedError#analysis}). A bare
   * aspect key, and any key type other than {@link ConfiguredTargetKey}, gets nothing.
   */
  private static void maybePostFailureEvents(
      ExtendedEventHandler eventHandler,
      @Nullable EventBus eventBus,
      boolean inBuildViewTest,
      ClassifiedError error) {
    if (inBuildViewTest) {
      // eventBus is null, but tests can still assert on the expected root causes being found.
      eventHandler.handle(Event.error(error.analysisRootCauses().toList().toString()));
      return;
    }

    Preconditions.checkNotNull(eventBus);
    // AnalysisFailureEvent.whileAnalyzingTarget can only name a configured target, so a failing
    // aspect is reported against the configured target it was applied to.
    if (error.normalizedKey() instanceof TopLevelAspectsKey topLevelAspectsKey) {
      if (error.isAnalysisError()) {
        eventBus.post(
            AnalysisFailureEvent.whileAnalyzingTarget(
                topLevelAspectsKey.getBaseConfiguredTargetKey(), error.analysisRootCauses()));
      }
      return;
    }
    if (!(error.normalizedKey() instanceof ConfiguredTargetKey ctKey)) {
      return;
    }

    // For loading errors, we expect both LoadingFailureEvent and AnalysisFailureEvent.
    if (error.isLoadingError()) {
      for (Label loadingRootCause : error.loadingRootCauses()) {
        // This event is only for backwards compatibility with the old event protocol. Remove
        // once we've migrated to the build event protocol.
        eventBus.post(
            new LoadingFailureEvent(Preconditions.checkNotNull(error.label()), loadingRootCause));
      }
    }

    if (error.isAnalysisError()) {
      eventBus.post(AnalysisFailureEvent.whileAnalyzingTarget(ctKey, error.analysisRootCauses()));
    }
  }

  /**
   * Aborts a {@code --nokeep_going} build with the error that {@link #NO_KEEP_GOING_PRECEDENCE}
   * picked out.
   *
   * <p>Action conflicts are harvested before classification and never reach this method.
   *
   * @throws ViewCreationFailedException when the root cause is analysis-related.
   * @throws BuildFailedException when the root cause is execution-related.
   * @throws TestExecException when the root cause is test-related.
   */
  private static void abortBuild(
      EvaluationResult<? extends SkyValue> result, BugReporter bugReporter, ClassifiedError error)
      throws BuildFailedException, TestExecException, ViewCreationFailedException {
    // If the error is execution-related: straightaway rethrow. No further steps required.
    if (error.reportingPriority() == ReportingPriority.EXECUTION) {
      if (error.isCycle()) {
        // A --nokeep_going build that found a cycle had no other error to throw (otherwise it
        // wouldn't have bothered looking for a cycle), and the cycle itself has already been
        // reported, so a generic build failure is the best we can do. Analysis cycles are handled
        // below.
        throw new BuildFailedException(null, CYCLE_CODE);
      }
      rethrow(error.cause(), bugReporter, result);
    }

    throw createViewCreationFailedException(
        error.cause(),
        error.reportingPriority() == ReportingPriority.ASPECT_ANALYSIS
            ? String.format(
                "Analysis of aspects '%s' failed; build aborted",
                describeAspect((AspectBaseKey) error.normalizedKey()))
            : String.format("Analysis of target '%s' failed; build aborted", error.label()));
  }

  private static String describeAspect(AspectBaseKey aspectKey) {
    return aspectKey instanceof TopLevelAspectsKey topLevelAspectsKey
        ? topLevelAspectsKey.getDescription()
        : ((AspectKey) aspectKey).prettyPrint();
  }

  /**
   * Classifies one single error from the result.
   *
   * <p>No exception is ever thrown here, and nothing is posted: this only gathers the information
   * around one single error. {@link #processErrors} decides what to do with it.
   */
  private static ClassifiedError classify(
      EvaluationResult<? extends SkyValue> result,
      BugReporter bugReporter,
      SkyKey errorKey,
      ErrorInfo errorInfo) {
    Exception cause = errorInfo.getException();

    if (errorKey instanceof ActionLookupData || isExecutionException(cause)) {
      return ClassifiedError.execution(
          errorKey,
          cause,
          getExecutionDetailedExitCodeFromCause(result, cause, bugReporter),
          cause instanceof ActionExecutionException actionExecutionException
              ? actionExecutionException.getRootCauses()
              : NestedSetBuilder.emptySet(Order.STABLE_ORDER));
    }

    ActionLookupKey actionLookupKey =
        switch (errorKey) {
          case ActionLookupKey key -> key;
          default -> throw new IllegalStateException("Unexpected error key: " + errorKey);
        };

    // A cycle is the only way to fail without an exception.
    if (cause == null) {
      return isExecutionCycle(errorInfo.getCycleInfo())
          ? ClassifiedError.execution(
              errorKey, null, CYCLE_CODE, NestedSetBuilder.emptySet(Order.STABLE_ORDER))
          : ClassifiedError.analysis(errorKey, null, cycleRootCauses(actionLookupKey, errorInfo));
    }

    NestedSet<Cause> analysisRootCauses =
        switch (cause) {
          case ConfiguredValueCreationException e -> e.getRootCauses();
          case AspectCreationException e -> e.getCauses();
          // Reported entirely through the DetailedExitCode; no label is implicated.
          case TargetCompatibilityCheckException _ -> NestedSetBuilder.emptySet(Order.STABLE_ORDER);
          // This arm must come last: SaneAnalysisException extends DetailedException, so every
          // well-behaved analysis exception that does not carry root causes of its own lands here.
          // In a --keep_going build most of them arrive wrapped in a
          // ConfiguredValueCreationException instead, which does carry them.
          case DetailedException e ->
              NestedSetBuilder.create(
                  Order.STABLE_ORDER,
                  new AnalysisFailedCause(
                      actionLookupKey.getLabel(),
                      configurationIdMessage(actionLookupKey.getConfigurationKey()),
                      e.getDetailedExitCode()));
          default -> {
            BugReport.logUnexpected(
                cause, "Unexpected cause encountered while evaluating: %s", errorKey);
            yield NestedSetBuilder.emptySet(Order.STABLE_ORDER);
          }
        };
    return ClassifiedError.analysis(errorKey, cause, analysisRootCauses);
  }

  /** The root causes of an analysis cycle: the culprit, if we can name one. */
  private static NestedSet<Cause> cycleRootCauses(ActionLookupKey errorKey, ErrorInfo errorInfo) {
    Label culprit =
        maybeGetConfiguredTargetCycleCulprit(errorKey.getLabel(), errorInfo.getCycleInfo());
    // TODO(ulfjack): We need to report the dependency cycle here. How?
    return culprit == null
        ? NestedSetBuilder.emptySet(Order.STABLE_ORDER)
        : NestedSetBuilder.create(
            Order.STABLE_ORDER,
            new LabelCause(
                culprit, DetailedExitCode.of(createFailureDetail("Dependency cycle", Code.CYCLE))));
  }

  private static DetailedExitCode getExecutionDetailedExitCodeFromCause(
      EvaluationResult<? extends SkyValue> result, Exception cause, BugReporter bugReporter) {
    DetailedExitCode executionDetailedExitCode = DetailedException.getDetailedExitCode(cause);
    if (executionDetailedExitCode == null) {
      executionDetailedExitCode =
          sendBugReportAndCreateUnknownExecutionDetailedExitCode(result, cause, bugReporter);
    }
    return executionDetailedExitCode;
  }

  private static DetailedExitCode sendBugReportAndCreateUnknownExecutionDetailedExitCode(
      EvaluationResult<? extends SkyValue> result, Throwable cause, BugReporter bugReporter) {
    // An undetailed exception means we may incorrectly attribute responsibility for the failure:
    // we need to fix that.
    bugReporter.sendNonFatalBugReport(
        new IllegalStateException(
            "action terminated with unexpected exception with result " + result, cause));
    String message =
        "Unexpected exception, please file an issue with the Bazel team: " + cause.getMessage();
    return createDetailedExecutionExitCode(message, UNKNOWN_EXECUTION);
  }

  private static void logOrPrintWarningsKeepGoing(
      ClassifiedError error, ExtendedEventHandler eventHandler) {
    if (error.reportingPriority() == ReportingPriority.EXECUTION) {
      // The action machinery has already told the user about this, so only log the exceptions that
      // it doesn't know about.
      if (error.cause() != null && isExecutionCauseWorthLogging(error.cause())) {
        logger.atWarning().withCause(error.cause()).log(
            "Non-action-execution/input-error exception while building target %s", error.label());
      }
      return;
    }
    var message =
        String.format(
            "errors encountered while analyzing target '%s', it will not be built.", error.label());
    if (error.cause() != null) {
      message += String.format("\n%s", error.cause().getMessage());
    }
    eventHandler.handle(Event.warn(message));
  }

  private static boolean isExecutionCauseWorthLogging(Throwable cause) {
    return !(cause instanceof ActionExecutionException)
        && !(cause instanceof InputFileErrorException)
        && !(cause instanceof TopLevelOutputException);
  }

  private static boolean isValidErrorKeyType(SkyKey errorKey) {
    return errorKey instanceof ConfiguredTargetKey || errorKey instanceof AspectBaseKey;
  }

  /** Peel away the wrapper layers to get to the ActionLookupKey of the top level target. */
  private static SkyKey getEffectiveErrorKey(SkyKey key) {
    return switch (key) {
      case BuildDriverKey buildDriverKey -> buildDriverKey.getActionLookupKey();
      // For exclusive tests.
      case TestCompletionKey testCompletionKey -> testCompletionKey.configuredTargetKey();
      // For non-skymeld action executions.
      case TargetCompletionKey targetCompletionKey -> targetCompletionKey.actionLookupKey();
      case AspectCompletionKey aspectCompletionKey -> aspectCompletionKey.actionLookupKey();
      default -> key;
    };
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
      ErrorInfo errorInfo, SkyKey key, WalkableGraph walkableGraph, boolean keepEdges)
      throws InterruptedException {
    Throwable cause = errorInfo.getException();
    if (cause == null) {
      // Cycle.
      return;
    }

    if (convertToAnalysisException(cause) != null) {
      // Valid exception type.
      return;
    }

    logUnexpectedExceptionOrigin(errorInfo, key, walkableGraph, cause, keepEdges);
  }

  private static void assertValidAnalysisOrExecutionException(
      ErrorInfo errorInfo, SkyKey key, WalkableGraph walkableGraph, boolean keepEdges)
      throws InterruptedException {
    Throwable cause = errorInfo.getException();
    if (cause == null) {
      // Cycle.
      return;
    }

    if (convertToAnalysisException(cause) != null || isExecutionException(cause)) {
      // Valid exception type.
      return;
    }

    logUnexpectedExceptionOrigin(errorInfo, key, walkableGraph, cause, keepEdges);
  }

  /**
   * Walk the graph to find a path to the lowest-level node that threw unexpected exception and log
   * it.
   */
  private static void logUnexpectedExceptionOrigin(
      ErrorInfo errorInfo,
      SkyKey key,
      WalkableGraph walkableGraph,
      Throwable cause,
      boolean keepEdges)
      throws InterruptedException {
    if (!keepEdges) {
      // Can't traverse the graph to find the origin.
      logUnexpectedException(key, errorInfo, "direct deps not stored");
      return;
    }
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
      logUnexpectedException(key, errorInfo, path);
    }
  }

  private static void logUnexpectedException(SkyKey key, ErrorInfo errorInfo, Object extraInfo) {
    BugReport.logUnexpected("Unexpected analysis error: %s -> %s, (%s)", key, errorInfo, extraInfo);
  }

  @Nullable
  private static DetailedException convertToAnalysisException(Throwable cause) {
    // The cause may be NoSuch{Target,Package}Exception if we run the reduced loading phase and then
    // analyze with --nokeep_going.
    if (cause instanceof SaneAnalysisException
        || cause instanceof NoSuchTargetException
        || cause instanceof NoSuchPackageException
        || cause instanceof ExternalDepsException) {
      return (DetailedException) cause;
    }
    return null;
  }

  private static boolean isExecutionException(@Nullable Throwable cause) {
    return cause instanceof ActionExecutionException
        || cause instanceof InputFileErrorException
        || cause instanceof TestExecException
        // Refer to UnusedInputsFailureIntegrationTest#incrementalFailureOnUnusedInput.
        || cause instanceof ArtifactNestedSetEvalException
        // For top-level outputs errors in CompletionFunction.
        || cause instanceof TopLevelOutputException;
  }

  /**
   * Figure out why an action's analysis/execution failed and rethrow the right kind of exception.
   */
  @VisibleForTesting
  static void rethrow(
      Throwable cause, BugReporter bugReporter, EvaluationResult<?> resultForDebugging)
      throws BuildFailedException, TestExecException {
    Throwables.throwIfUnchecked(cause);
    Throwable innerCause = cause.getCause();
    if (innerCause instanceof TestExecException testExecException) {
      throw testExecException;
    }
    if (cause instanceof ActionExecutionException actionExecutionCause) {
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
          /* errorAlreadyShown= */ !actionExecutionCause.showError(),
          actionExecutionCause.getDetailedExitCode());
    }
    if (cause instanceof InputFileErrorException inputFileErrorException) {
      throw inputFileErrorException;
    }
    if (cause instanceof TopLevelOutputException topLevelOutputException) {
      throw topLevelOutputException;
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

    DetailedExitCode unknownExitCode =
        sendBugReportAndCreateUnknownExecutionDetailedExitCode(
            resultForDebugging, cause, bugReporter);
    throw new BuildFailedException(
        Preconditions.checkNotNull(unknownExitCode.getFailureDetail()).getMessage(),
        unknownExitCode);
  }

  private static final DetailedExitCode CYCLE_CODE =
      createDetailedExecutionExitCode("cycle found during execution", Execution.Code.CYCLE);
  private static final Execution UNKNOWN_EXECUTION =
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

  private static boolean isExecutionCycle(Iterable<CycleInfo> cycleInfoCollection) {
    for (CycleInfo cycleInfo : cycleInfoCollection) {
      if (cycleInfo.getCycle().stream().allMatch(ACTION_OR_ARTIFACT_OR_TRANSITIVE_RDEP)) {
        // All these cycle info belong to the same top level key. If one of them is
        // execution-related, we consider the error to be execution-related.
        return true;
      }
    }
    return false;
  }
}
