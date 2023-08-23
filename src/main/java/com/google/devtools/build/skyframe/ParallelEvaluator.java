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
package com.google.devtools.build.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.flogger.GoogleLogger;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.EvaluationContext.UnnecessaryTemporaryStateDropperReceiver;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationSuccessState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * This class is not intended for direct use, and is only exposed as public for use in evaluation
 * implementations outside of this package.
 *
 * <p>Note on naming: there used to be an {@code Evaluator} interface this class (and likely some
 * others) implemented, but as of 2020-01-15 this was the only implementation so we deleted that
 * interface. Now {@code ParallelEvaluator} could be called just {@code Evaluator}, but renaming it
 * is not worth the effort.
 */
public class ParallelEvaluator extends AbstractParallelEvaluator {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();
  private final UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver;

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      Version minimalVersion,
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      QuiescingExecutor executor,
      CycleDetector cycleDetector,
      boolean mergingSkyframeAnalysisExecutionPhases,
      UnnecessaryTemporaryStateDropperReceiver unnecessaryTemporaryStateDropperReceiver) {
    super(
        graph,
        graphVersion,
        minimalVersion,
        skyFunctions,
        reporter,
        emittedEventState,
        storedEventFilter,
        errorInfoManager,
        keepGoing,
        progressReceiver,
        graphInconsistencyReceiver,
        executor,
        cycleDetector,
        mergingSkyframeAnalysisExecutionPhases);
    this.unnecessaryTemporaryStateDropperReceiver = unnecessaryTemporaryStateDropperReceiver;
  }

  private void informProgressReceiverThatValueIsDone(SkyKey key, NodeEntry entry)
      throws InterruptedException {
    if (evaluatorContext.getProgressReceiver() == null) {
      return;
    }
    Preconditions.checkState(entry.isDone(), entry);
    SkyValue value = entry.getValue();
    Version valueVersion = entry.getVersion();
    Preconditions.checkState(
        valueVersion.atMost(evaluatorContext.getGraphVersion()),
        "%s should be at most %s in the version partial ordering",
        valueVersion,
        evaluatorContext.getGraphVersion());

    ErrorInfo error = null;
    SkyValue valueMaybeWithMetadata = entry.getValueMaybeWithMetadata();
    if (valueMaybeWithMetadata != null) {
      replay(ValueWithMetadata.wrapWithMetadata(valueMaybeWithMetadata));
      error = ValueWithMetadata.getMaybeErrorInfo(valueMaybeWithMetadata);
    }

    // For most nodes we do not inform the progress receiver if they were already done when we
    // retrieve them, but top-level nodes are presumably of more interest.
    // If valueVersion is not equal to graphVersion, it must be less than it (by the
    // Preconditions check above), and so the node is clean.
    EvaluationState evaluationState =
        valueVersion.equals(evaluatorContext.getGraphVersion())
            ? EvaluationState.BUILT
            : EvaluationState.CLEAN;
    evaluatorContext
        .getProgressReceiver()
        .evaluated(
            key,
            evaluationState == EvaluationState.BUILT ? value : null,
            evaluationState == EvaluationState.BUILT ? error : null,
            value != null
                ? EvaluationSuccessState.SUCCESS.supplier()
                : EvaluationSuccessState.FAILURE.supplier(),
            evaluationState,
            /* directDeps= */ null);
  }

  @ThreadCompatible
  private <T extends SkyValue> EvaluationResult<T> doMutatingEvaluation(
      ImmutableSet<SkyKey> skyKeys) throws InterruptedException {
    injectErrorTransienceValue();
    try {
      NodeBatch batch = graph.createIfAbsentBatch(null, Reason.PRE_OR_POST_EVALUATION, skyKeys);
      for (SkyKey skyKey : skyKeys) {
        NodeEntry entry = batch.get(skyKey);
        // This must be equivalent to the code in AbstractParallelEvaluator.Evaluate#enqueueChild,
        // in order to be thread-safe.
        switch (entry.addReverseDepAndCheckIfDone(null)) {
          case NEEDS_SCHEDULING:
            evaluatorContext.getVisitor().enqueueEvaluation(skyKey, entry.getPriority(), null);
            break;
          case DONE:
            informProgressReceiverThatValueIsDone(skyKey, entry);
            break;
          case ALREADY_EVALUATING:
            break;
          default:
            throw new IllegalStateException(entry + " for " + skyKey + " in unknown state");
        }
      }
    } catch (InterruptedException ie) {
      // When multiple keys are being evaluated, it's possible that a key may get queued before
      // an InterruptedException is thrown from either #addReverseDepAndCheckIfDone or
      // #informProgressReceiverThatValueIsDone on a different key. Therefore we have to make sure
      // all evaluation threads are properly interrupted and shut down, if main thread (current
      // thread) is interrupted.
      Thread.currentThread().interrupt();
      try {
        evaluatorContext.getVisitor().waitForCompletion();
      } catch (SchedulerException se) {
        // A SchedulerException due to a SkyFunction observing the interrupt is completely expected.
        if (!(se.getCause() instanceof InterruptedException)) {
          throw se;
        }
      }

      // Rethrow the InterruptedException to avoid proceeding to construct the result.
      throw ie;
    }

    return waitForCompletionAndConstructResult(skyKeys);
  }

  private void injectErrorTransienceValue() throws InterruptedException {
    // We unconditionally add the ErrorTransienceValue here, to ensure that it will be created, and
    // in the graph, by the time that it is needed. Creating it on demand in a parallel context sets
    // up a race condition, because there is no way to atomically create a node and set its value.
    NodeEntry errorTransienceEntry =
        graph
            .createIfAbsentBatch(
                null, Reason.PRE_OR_POST_EVALUATION, ImmutableList.of(ErrorTransienceValue.KEY))
            .get(ErrorTransienceValue.KEY);
    if (!errorTransienceEntry.isDone()) {
      injectValues(
          ImmutableMap.of(ErrorTransienceValue.KEY, Delta.justNew(ErrorTransienceValue.INSTANCE)),
          evaluatorContext.getGraphVersion(),
          graph,
          evaluatorContext.getProgressReceiver());
    }
  }

  private <T extends SkyValue> EvaluationResult<T> waitForCompletionAndConstructResult(
      Iterable<SkyKey> skyKeys) throws InterruptedException {
    Map<SkyKey, ValueWithMetadata> bubbleErrorInfo = null;
    boolean catastrophe = false;
    try {
      evaluatorContext.getVisitor().waitForCompletion();
    } catch (final SchedulerException e) {
      propagateEvaluatorContextCrashIfAny();
      propagateInterruption(e);
      SkyKey errorKey = Preconditions.checkNotNull(e.getFailedValue(), e);
      // ErrorInfo could only be null if SchedulerException wrapped an InterruptedException, but
      // that should have been propagated.
      ErrorInfo errorInfo = Preconditions.checkNotNull(e.getErrorInfo(), errorKey);
      bubbleErrorInfo = bubbleErrorUp(errorInfo, errorKey, skyKeys, e.getRdepsToBubbleUpTo());
      if (evaluatorContext.keepGoing()) {
        Preconditions.checkState(
            errorInfo.isCatastrophic(),
            "Scheduler exception only thrown for catastrophe in keep_going evaluation: %s",
            e);
        catastrophe = true;
        // For b/287183296
        logger.atInfo().withCause(e).log(
            "Catastrophic exception in --keep_going mode while evaluating SkyKey: %s", errorKey);
      }
    }
    Preconditions.checkState(
        evaluatorContext.getVisitor().getCrashes().isEmpty(),
        evaluatorContext.getVisitor().getCrashes());

    // Successful evaluation, barring evaluation-wide exceptions, either because keepGoing or
    // because we actually did succeed.
    // TODO(bazel-team): Maybe report root causes during the build for lower latency.
    return constructResult(skyKeys, bubbleErrorInfo, catastrophe);
  }

  /**
   * Walk up graph to find a top-level node (without parents) that wanted this failure. Store the
   * failed nodes along the way in a map, with ErrorInfos that are appropriate for that layer.
   * Example:
   *
   * <pre>
   *                      foo   bar
   *                        \   /
   *           unrequested   baz
   *                     \    |
   *                      failed-node
   * </pre>
   *
   * User requests foo, bar. When failed-node fails, we look at its parents. unrequested is not
   * in-flight, so we replace failed-node by baz and repeat. We look at baz's parents. foo is
   * in-flight, so we replace baz by foo. Since foo is a top-level node and doesn't have parents, we
   * then break, since we know a top-level node, foo, that depended on the failed node.
   *
   * <p>There's the potential for a weird "track jump" here in the case:
   *
   * <pre>
   *                        foo
   *                       / \
   *                   fail1 fail2
   * </pre>
   *
   * If fail1 and fail2 fail simultaneously, fail2 may start propagating up in the loop below.
   * However, foo requests fail1 first, and then throws an exception based on that. This is not
   * incorrect, but may be unexpected.
   *
   * <p>Returns a map of errors that have been constructed during the bubbling up, so that the
   * appropriate error can be returned to the caller, even though that error was not written to the
   * graph. If a cycle is detected during the bubbling, this method aborts and returns null so that
   * the normal cycle detection can handle the cycle.
   *
   * <p>Note that we are not propagating error to the first top-level node but to the highest one,
   * because during this process we can add useful information about error from other nodes.
   *
   * <p>Every node on this walk but the leaf node is not done, by the following argument: the leaf
   * node is done, but the parents of it that we consider are in {@code rdepsToBubbleUpTo}. Each
   * parent is either (1) a parent that requested the leaf node and found it to be in error, meaning
   * it is not done, or (2) a parent that had registered a dependency on this leaf node before it
   * finished building. In the second case, that parent would not have been enqueued, since we
   * failed fast and prevented all new evaluations. Thus, we will only visit unfinished parents of
   * the leaf node. For the inductive argument, the only parents we consider are those that were
   * registered during this build (via {@link NodeEntry#getInProgressReverseDeps}. Since we don't
   * allow a node to build with unfinished deps, those parents cannot have built.
   */
  @SuppressWarnings("LenientFormatStringValidation")
  @Nullable
  private Map<SkyKey, ValueWithMetadata> bubbleErrorUp(
      final ErrorInfo leafFailure,
      SkyKey errorKey,
      Iterable<SkyKey> roots,
      Set<SkyKey> rdepsToBubbleUpTo)
      throws InterruptedException {
    // Remove all the compute states so as to give the SkyFunctions a chance to do fresh
    // computations during error bubbling.
    stateCache.invalidateAll();

    Set<SkyKey> rootValues = ImmutableSet.copyOf(roots);
    ErrorInfo error = leafFailure;
    LinkedHashMap<SkyKey, ValueWithMetadata> bubbleErrorInfo = new LinkedHashMap<>();
    boolean externalInterrupt = false;
    boolean firstIteration = true;
    while (true) {
      NodeEntry errorEntry =
          Preconditions.checkNotNull(graph.get(null, Reason.ERROR_BUBBLING, errorKey), errorKey);
      Iterable<SkyKey> reverseDeps;
      if (errorEntry.isDone()) {
        Preconditions.checkState(
            firstIteration,
            "Non-leaf done node reached: %s %s %s %s %s",
            errorKey,
            leafFailure,
            roots,
            rdepsToBubbleUpTo,
            bubbleErrorInfo);
        reverseDeps = rdepsToBubbleUpTo;
      } else {
        Preconditions.checkState(
            !firstIteration,
            "undone first iteration: %s %s %s %s %s %s",
            errorKey,
            errorEntry,
            leafFailure,
            roots,
            rdepsToBubbleUpTo,
            bubbleErrorInfo);
        reverseDeps = errorEntry.getInProgressReverseDeps();
      }
      firstIteration = false;
      // We should break from loop only when node doesn't have any parents.
      if (Iterables.isEmpty(reverseDeps)) {
        Preconditions.checkState(
            rootValues.contains(errorKey),
            "Current key %s has to be a top-level key: %s",
            errorKey,
            rootValues);
        SkyValue valueMaybeWithMetadata = errorEntry.getValueMaybeWithMetadata();
        if (valueMaybeWithMetadata != null) {
          replay(ValueWithMetadata.wrapWithMetadata(valueMaybeWithMetadata));
        }
        break;
      }
      SkyKey parent = Preconditions.checkNotNull(Iterables.getFirst(reverseDeps, null));
      if (bubbleErrorInfo.containsKey(parent)) {
        logger.atInfo().log(
            "Bubbled into a cycle. Don't try to bubble anything up. Cycle detection will kick in."
                + " %s: %s, %s, %s, %s, %s",
            parent, errorEntry, bubbleErrorInfo, leafFailure, roots, rdepsToBubbleUpTo);
        return null;
      }
      NodeEntry parentEntry =
          Preconditions.checkNotNull(
              graph.get(errorKey, Reason.ERROR_BUBBLING, parent),
              "parent %s of %s not in graph",
              parent,
              errorKey);
      Preconditions.checkState(
          !parentEntry.isDone(),
          "We cannot bubble into a done node entry: a done node cannot depend on a not-done node,"
              + " and the first errorParent was not done: %s %s %s %s %s %s %s %s",
          errorKey,
          errorEntry,
          parent,
          parentEntry,
          leafFailure,
          roots,
          rdepsToBubbleUpTo,
          bubbleErrorInfo);
      // Expected 6 args, but got 8.
      Preconditions.checkState(
          evaluatorContext.getProgressReceiver().isInflight(parent),
          "In-progress reverse deps can only include in-flight nodes: " + "%s %s %s %s %s %s",
          errorKey,
          errorEntry,
          parent,
          parentEntry,
          leafFailure,
          roots,
          rdepsToBubbleUpTo,
          bubbleErrorInfo);
      // Expected 6 args, but got 8.
      Preconditions.checkState(
          parentEntry.getTemporaryDirectDeps().contains(errorKey),
          "In-progress reverse deps can only include nodes that have declared a dep: "
              + "%s %s %s %s %s %s",
          errorKey,
          errorEntry,
          parent,
          parentEntry,
          leafFailure,
          roots,
          rdepsToBubbleUpTo,
          bubbleErrorInfo);
      Preconditions.checkNotNull(parentEntry, "%s %s", errorKey, parent);
      SkyFunction skyFunction = evaluatorContext.getSkyFunctions().get(parent.functionName());
      if (parentEntry.isDirty()) {
        switch (parentEntry.getDirtyState()) {
          case CHECK_DEPENDENCIES:
            // If this value's child was bubbled up to, it did not signal this value, and so we must
            // manually make it ready to build.
            parentEntry.signalDep(evaluatorContext.getGraphVersion(), errorKey);
            // Fall through to NEEDS_REBUILDING, since state is now NEEDS_REBUILDING.
          case NEEDS_REBUILDING:
            maybeMarkRebuilding(parentEntry);
            break;
          case NEEDS_FORCED_REBUILDING:
            parentEntry.forceRebuild();
            break;
          case REBUILDING:
            break;
          default:
            throw new AssertionError(parent + " not in valid dirty state: " + parentEntry);
        }
      }
      SkyKey childErrorKey = errorKey;
      errorKey = parent;
      SkyFunctionEnvironment env =
          SkyFunctionEnvironment.createForError(
              parent,
              parentEntry.getTemporaryDirectDeps(),
              bubbleErrorInfo,
              ImmutableSet.of(),
              evaluatorContext);
      externalInterrupt = externalInterrupt || Thread.currentThread().isInterrupted();
      boolean completedRun = false;
      try {
        // This build is only to check if the parent node can give us a better error. We don't
        // care about a return value.
        skyFunction.compute(parent, env);
        completedRun = true;
      } catch (InterruptedException interruptedException) {
        logger.atInfo().withCause(interruptedException).log("Interrupted during %s eval", parent);
        // Do nothing.
        // This throw happens if the builder requested the failed node, and then checked the
        // interrupted state later -- getValueOrThrow sets the interrupted bit after the failed
        // value is requested, to prevent the builder from doing too much work.
      } catch (SkyFunctionException builderException) {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
        ReifiedSkyFunctionException reifiedBuilderException =
            new ReifiedSkyFunctionException(builderException);
        error =
            ErrorInfo.fromException(reifiedBuilderException, /*isTransitivelyTransient=*/ false);
        NestedSet<Reportable> events =
            env.reportEventsAndGetEventsToStore(parentEntry, /*expectDoneDeps=*/ false);
        ValueWithMetadata valueWithMetadata =
            ValueWithMetadata.error(
                ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)), events);
        replay(valueWithMetadata);
        bubbleErrorInfo.put(errorKey, valueWithMetadata);
        continue;
      } catch (RuntimeException e) {
        // About to crash. Print debugging to INFO log.
        logger.atSevere().log("Crashing on %s. Contents of bubbleErrorInfo:", parent);
        for (Map.Entry<SkyKey, ValueWithMetadata> bubbleEntry : bubbleErrorInfo.entrySet()) {
          logger.atSevere().log(
              "  %.1000s -> %.1000s", bubbleEntry.getKey(), bubbleEntry.getValue());
        }
        throw e;
      } finally {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
      }
      // TODO(b/166268889, b/172223413): remove when fixed.
      if (completedRun
          && error.getException() != null
          && (error.getException() instanceof IOException
              || error.getException().getClass().getName().endsWith("SourceArtifactException"))) {
        String skyFunctionName = parent.functionName().getName();
        if (!skyFunctionName.startsWith("FILE")
            && !skyFunctionName.startsWith("DIRECTORY_LISTING")) {
          logger.atInfo().log(
              "SkyFunction did not rethrow error, may be a bug that it did not expect one: %s"
                  + " via %s, %s (%s)",
              errorKey, childErrorKey, error, bubbleErrorInfo);
        }
      }
      if (completedRun && !env.encounteredErrorDuringBubbling()) {
        logger.atInfo().log(
            "Skyfunction did not encounter error: %s via %s, %s (%s)",
            errorKey, childErrorKey, error, bubbleErrorInfo);
      }
      // Builder didn't throw its own exception, so just propagate this one up.
      NestedSet<Reportable> events =
          env.reportEventsAndGetEventsToStore(parentEntry, /*expectDoneDeps=*/ false);
      ValueWithMetadata valueWithMetadata =
          ValueWithMetadata.error(
              ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)), events);
      replay(valueWithMetadata);
      bubbleErrorInfo.put(errorKey, valueWithMetadata);
    }

    // Reset the interrupt bit if there was an interrupt from outside this evaluator interrupt.
    // Note that there are internal interrupts set in the node builder environment if an error
    // bubbling node calls getValueOrThrow() on a node in error.
    if (externalInterrupt) {
      Thread.currentThread().interrupt();
    }
    return bubbleErrorInfo;
  }

  /**
   * Constructs an {@link EvaluationResult} from the {@link #graph}. Looks for cycles if there are
   * unfinished nodes but no error was already found through bubbling up (as indicated by {@code
   * bubbleErrorInfo} being null).
   *
   * <p>{@code visitor} may be null, but only in the case where all graph entries corresponding to
   * {@code skyKeys} are known to be in the DONE state ({@code entry.isDone()} returns true).
   */
  private <T extends SkyValue> EvaluationResult<T> constructResult(
      Iterable<SkyKey> skyKeys,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      boolean catastrophe)
      throws InterruptedException {
    Preconditions.checkState(
        !catastrophe || evaluatorContext.keepGoing(),
        "Catastrophe not consistent with keepGoing mode: %s %s %s",
        skyKeys,
        catastrophe,
        bubbleErrorInfo);
    EvaluationResult.Builder<T> result = EvaluationResult.builder();
    List<SkyKey> cycleRoots = new ArrayList<>();
    boolean haveKeys = false;
    for (SkyKey skyKey : skyKeys) {
      haveKeys = true;
      SkyValue unwrappedValue =
          maybeGetValueFromError(
              skyKey, graph.get(null, Reason.PRE_OR_POST_EVALUATION, skyKey), bubbleErrorInfo);
      ValueWithMetadata valueWithMetadata =
          unwrappedValue == null ? null : ValueWithMetadata.wrapWithMetadata(unwrappedValue);
      // Cycle checking: if there is a cycle, evaluation cannot progress, therefore,
      // the final values will not be in DONE state when the work runs out.
      if (valueWithMetadata == null) {
        // Don't look for cycles if the build failed for a known reason.
        if (bubbleErrorInfo == null) {
          cycleRoots.add(skyKey);
        }
        continue;
      }
      SkyValue value = valueWithMetadata.getValue();
      ErrorInfo errorInfo = valueWithMetadata.getErrorInfo();
      Preconditions.checkState(value != null || errorInfo != null, skyKey);
      if (!evaluatorContext.keepGoing() && errorInfo != null) {
        // value will be null here unless the value was already built on a prior keepGoing build.
        result.addError(skyKey, errorInfo);
        continue;
      }
      if (value == null) {
        // Note that we must be in the keepGoing case. Only make this value an error if it doesn't
        // have a value. The error shouldn't matter to the caller since the value succeeded after a
        // fashion.
        result.addError(skyKey, errorInfo);
      } else {
        result.addResult(skyKey, value);
      }
    }
    if (!cycleRoots.isEmpty()) {
      logger.atInfo().log("Detecting cycles with roots: %s", cycleRoots);
      cycleDetector.checkForCycles(cycleRoots, result, evaluatorContext);
    }
    Preconditions.checkState(
        !result.isEmpty() || !haveKeys,
        "No result for keys %s (%s %s)",
        skyKeys,
        bubbleErrorInfo,
        catastrophe);
    result.maybeEnsureCatastrophe(catastrophe);
    EvaluationResult<T> builtResult = result.build();
    Preconditions.checkState(
        bubbleErrorInfo == null || builtResult.hasError(),
        "If an error bubbled up, some top-level node must be in error: %s %s %s",
        bubbleErrorInfo,
        skyKeys,
        builtResult);
    return builtResult;
  }

  @Nullable
  private static SkyValue maybeGetValueFromError(
      SkyKey key,
      @Nullable NodeEntry entry,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo)
      throws InterruptedException {
    SkyValue value = bubbleErrorInfo == null ? null : bubbleErrorInfo.get(key);
    if (value != null) {
      return value;
    }
    return isDoneForBuild(entry) ? entry.getValueMaybeWithMetadata() : null;
  }

  static void injectValues(
      Map<SkyKey, Delta> injectionMap,
      Version version,
      ProcessableGraph graph,
      DirtyTrackingProgressReceiver progressReceiver)
      throws InterruptedException {
    NodeBatch prevNodeEntries =
        graph.createIfAbsentBatch(null, Reason.OTHER, injectionMap.keySet());
    for (Map.Entry<SkyKey, Delta> injectionEntry : injectionMap.entrySet()) {
      SkyKey key = injectionEntry.getKey();
      SkyValue value = injectionEntry.getValue().newValue();
      NodeEntry prevEntry = prevNodeEntries.get(key);
      DependencyState newState = prevEntry.addReverseDepAndCheckIfDone(null);
      Preconditions.checkState(
          newState != DependencyState.ALREADY_EVALUATING, "%s %s", key, prevEntry);
      if (prevEntry.isDirty()) {
        // Get the node in the state where it is able to accept a value.
        Preconditions.checkState(
            newState == DependencyState.NEEDS_SCHEDULING, "%s %s", key, prevEntry);
        // If there was a node in the graph before, check that the previous node has no
        // dependencies. Overwriting a value with deps with an injected value (which is by
        // definition deps-free) needs a little additional bookkeeping (removing reverse deps from
        // the dependencies), but more importantly it's something that we want to avoid, because it
        // indicates confusion of input values and derived values.
        Preconditions.checkState(
            prevEntry.noDepsLastBuild(), "existing entry for %s has deps: %s", key, prevEntry);
      }
      prevEntry.markRebuilding();
      @Nullable
      Version maxTransitiveSourceVersion =
          injectionEntry.getValue().newMaxTransitiveSourceVersion();
      prevEntry.setValue(value, version, maxTransitiveSourceVersion);
      // Now that this key's injected value is set, it is no longer dirty.
      progressReceiver.injected(key);
    }
  }

  /**
   * Evaluates a set of values. Returns an {@link EvaluationResult}. All elements of skyKeys must be
   * keys for Values of subtype T.
   */
  @ThreadCompatible
  public <T extends SkyValue> EvaluationResult<T> eval(Iterable<? extends SkyKey> skyKeys)
      throws InterruptedException {
    ImmutableSet<SkyKey> skyKeySet = ImmutableSet.copyOf(skyKeys);

    // Optimization: if all required node values are already present in the cache, return them
    // directly without launching the heavy machinery, spawning threads, etc.
    // Inform progressReceiver that these nodes are done to be consistent with the main code path.
    boolean allAreDone = true;
    NodeBatch batch =
        evaluatorContext.getGraph().getBatch(null, Reason.PRE_OR_POST_EVALUATION, skyKeySet);
    for (SkyKey key : skyKeySet) {
      if (!isDoneForBuild(batch.get(key))) {
        allAreDone = false;
        break;
      }
    }
    if (allAreDone) {
      for (SkyKey skyKey : skyKeySet) {
        informProgressReceiverThatValueIsDone(skyKey, batch.get(skyKey));
      }
      // Note that the 'catastrophe' parameter doesn't really matter here (it's only used for
      // checking).
      return constructResult(skyKeySet, null, /*catastrophe=*/ false);
    }

    if (!evaluatorContext.keepGoing()) {
      Set<SkyKey> cachedErrorKeys = new HashSet<>();
      for (SkyKey skyKey : skyKeySet) {
        NodeEntry entry = graph.get(null, Reason.PRE_OR_POST_EVALUATION, skyKey);
        if (entry == null) {
          continue;
        }
        if (entry.isDone() && entry.getErrorInfo() != null) {
          informProgressReceiverThatValueIsDone(skyKey, entry);
          cachedErrorKeys.add(skyKey);
        }
      }

      // Errors, even cached ones, should halt evaluations not in keepGoing mode.
      if (!cachedErrorKeys.isEmpty()) {
        // Note that the 'catastrophe' parameter doesn't really matter here (it's only used for
        // checking).
        return constructResult(cachedErrorKeys, null, /*catastrophe=*/ false);
      }
    }

    unnecessaryTemporaryStateDropperReceiver.onEvaluationStarted(
        () -> evaluatorContext.stateCache().invalidateAll());
    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.SKYFRAME_EVAL, "Parallel Evaluator evaluation")) {
      return doMutatingEvaluation(skyKeySet);
    } finally {
      unnecessaryTemporaryStateDropperReceiver.onEvaluationFinished();
    }
  }


}
