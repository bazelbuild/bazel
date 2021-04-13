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
import com.google.devtools.build.lib.bugreport.BugReport;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationSuccessState;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
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
import java.util.concurrent.ExecutorService;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Evaluates a set of given functions ({@code SkyFunction}s) with arguments ({@code SkyKey}s).
 * Cycles are not allowed and are detected during the traversal.
 *
 * <p>This class implements multi-threaded evaluation. This is a fairly complex process that has
 * strong consistency requirements between the {@link ProcessableGraph}, the nodes in the graph of
 * type {@link NodeEntry}, the work queue, and the set of in-flight nodes.
 *
 * <p>The basic invariants are:
 *
 * <p>A node can be in one of three states: ready, waiting, and done. A node is ready if and only if
 * all of its dependencies have been signaled. A node is done if it has a value. It is waiting if
 * not all of its dependencies have been signaled.
 *
 * <p>A node must be in the work queue if and only if it is ready. It is an error for a node to be
 * in the work queue twice at the same time.
 *
 * <p>A node is considered in-flight if it has been created, and is not done yet. In case of an
 * interrupt, the work queue is discarded, and the in-flight set is used to remove partially
 * computed values.
 *
 * <p>Each evaluation of the graph takes place at a "version," which is currently given by a
 * non-negative {@code long}. The version can also be thought of as an "mtime." Each node in the
 * graph has a version, which is the last version at which its value changed. This version data is
 * used to avoid unnecessary re-evaluation of values. If a node is re-evaluated and found to have
 * the same data as before, its version (mtime) remains the same. If all of a node's children's have
 * the same version as before, its re-evaluation can be skipped.
 *
 * <p>Through its generic exception type parameter, this class supports a notion of an
 * "evaluation-wide" exception, to be implemented and detected by the classes that extend this one.
 * Those classes have the opportunity to check their conditions and throw instances of their
 * exception type in their {@link #bubbleErrorUpExceptionally} and {@link
 * #constructResultExceptionally} method implementations.
 */
public abstract class AbstractExceptionalParallelEvaluator<E extends Exception>
    extends AbstractParallelEvaluator {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  AbstractExceptionalParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      Supplier<ExecutorService> executorService,
      CycleDetector cycleDetector,
      EvaluationVersionBehavior evaluationVersionBehavior) {
    super(
        graph,
        graphVersion,
        skyFunctions,
        reporter,
        emittedEventState,
        storedEventFilter,
        errorInfoManager,
        keepGoing,
        progressReceiver,
        graphInconsistencyReceiver,
        executorService,
        cycleDetector,
        evaluationVersionBehavior);
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
            evaluationState);
  }

  <T extends SkyValue> EvaluationResult<T> evalExceptionally(Iterable<? extends SkyKey> skyKeys)
      throws InterruptedException, E {
    ImmutableSet<SkyKey> skyKeySet = ImmutableSet.copyOf(skyKeys);

    // Optimization: if all required node values are already present in the cache, return them
    // directly without launching the heavy machinery, spawning threads, etc.
    // Inform progressReceiver that these nodes are done to be consistent with the main code path.
    boolean allAreDone = true;
    Map<SkyKey, ? extends NodeEntry> batch =
        evaluatorContext.getBatchValues(null, Reason.PRE_OR_POST_EVALUATION, skyKeySet);
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
      return constructResultExceptionally(skyKeySet, null, /*catastrophe=*/ false);
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
        return constructResultExceptionally(cachedErrorKeys, null, /*catastrophe=*/ false);
      }
    }

    try (SilentCloseable c =
        Profiler.instance().profile(ProfilerTask.SKYFRAME_EVAL, "Parallel Evaluator evaluation")) {
      return doMutatingEvaluation(skyKeySet);
    }
  }

  @ThreadCompatible
  private <T extends SkyValue> EvaluationResult<T> doMutatingEvaluation(
      ImmutableSet<SkyKey> skyKeys) throws InterruptedException, E {
    injectErrorTransienceValue();
    try {
      for (Map.Entry<SkyKey, ? extends NodeEntry> e :
          graph.createIfAbsentBatch(null, Reason.PRE_OR_POST_EVALUATION, skyKeys).entrySet()) {
        SkyKey skyKey = e.getKey();
        NodeEntry entry = e.getValue();
        // This must be equivalent to the code in AbstractParallelEvaluator.Evaluate#enqueueChild,
        // in order to be thread-safe.
        switch (entry.addReverseDepAndCheckIfDone(null)) {
          case NEEDS_SCHEDULING:
            // Low priority because this node is not needed by any other currently evaluating node.
            // So keep it at the back of the queue as long as there's other useful work to be done.
            evaluatorContext.getVisitor().enqueueEvaluation(skyKey, Integer.MIN_VALUE);
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

  protected void injectErrorTransienceValue() throws InterruptedException {
    // We unconditionally add the ErrorTransienceValue here, to ensure that it will be created, and
    // in the graph, by the time that it is needed. Creating it on demand in a parallel context sets
    // up a race condition, because there is no way to atomically create a node and set its value.
    NodeEntry errorTransienceEntry =
        Iterables.getOnlyElement(
            graph
                .createIfAbsentBatch(
                    null, Reason.PRE_OR_POST_EVALUATION, ImmutableList.of(ErrorTransienceValue.KEY))
                .values());
    if (!errorTransienceEntry.isDone()) {
      injectValues(
          ImmutableMap.of(ErrorTransienceValue.KEY, ErrorTransienceValue.INSTANCE),
          evaluatorContext.getGraphVersion(),
          graph,
          evaluatorContext.getProgressReceiver());
    }
  }

  private <T extends SkyValue> EvaluationResult<T> waitForCompletionAndConstructResult(
      Iterable<SkyKey> skyKeys) throws InterruptedException, E {
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
      bubbleErrorInfo =
          bubbleErrorUpExceptionally(errorInfo, errorKey, skyKeys, e.getRdepsToBubbleUpTo());
      if (evaluatorContext.keepGoing()) {
        Preconditions.checkState(
            errorInfo.isCatastrophic(),
            "Scheduler exception only thrown for catastrophe in keep_going evaluation: %s",
            e);
        catastrophe = true;
      }
    }
    Preconditions.checkState(
        evaluatorContext.getVisitor().getCrashes().isEmpty(),
        evaluatorContext.getVisitor().getCrashes());

    // Successful evaluation, barring evaluation-wide exceptions, either because keepGoing or
    // because we actually did succeed.
    // TODO(bazel-team): Maybe report root causes during the build for lower latency.
    return constructResultExceptionally(skyKeys, bubbleErrorInfo, catastrophe);
  }

  abstract Map<SkyKey, ValueWithMetadata> bubbleErrorUpExceptionally(
      final ErrorInfo leafFailure,
      SkyKey errorKey,
      Iterable<SkyKey> roots,
      Set<SkyKey> rdepsToBubbleUpTo)
      throws InterruptedException, E;

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
  Map<SkyKey, ValueWithMetadata> bubbleErrorUp(
      final ErrorInfo leafFailure,
      SkyKey errorKey,
      Iterable<SkyKey> roots,
      Set<SkyKey> rdepsToBubbleUpTo)
      throws InterruptedException {
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
      Preconditions.checkState(
          parentEntry.getTemporaryDirectDeps().expensiveContains(errorKey),
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
      SkyFunction factory = evaluatorContext.getSkyFunctions().get(parent.functionName());
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
          case FORCED_REBUILDING:
            break;
          default:
            throw new AssertionError(parent + " not in valid dirty state: " + parentEntry);
        }
      }
      SkyKey childErrorKey = errorKey;
      errorKey = parent;
      SkyFunctionEnvironment env =
          new SkyFunctionEnvironment(
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
        factory.compute(parent, env);
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
        Pair<NestedSet<TaggedEvents>, NestedSet<Postable>> eventsAndPostables =
            env.buildAndReportEventsAndPostables(parentEntry, /*expectDoneDeps=*/ false);
        ValueWithMetadata valueWithMetadata =
            ValueWithMetadata.error(
                ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
                eventsAndPostables.first,
                eventsAndPostables.second);
        replay(valueWithMetadata);
        bubbleErrorInfo.put(errorKey, valueWithMetadata);
        continue;
      } finally {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
      }
      // TODO(b/166268889, b/172223413): remove when fixed.
      if (completedRun && error.getException() instanceof IOException) {
        logger.atInfo().log(
            "SkyFunction did not rethrow error, may be a bug that it did not expect one: %s"
                + " via %s, %s (%s)",
            errorKey, childErrorKey, error, bubbleErrorInfo);
      }
      // Builder didn't throw its own exception, so just propagate this one up.
      Pair<NestedSet<TaggedEvents>, NestedSet<Postable>> eventsAndPostables =
          env.buildAndReportEventsAndPostables(parentEntry, /*expectDoneDeps=*/ false);
      ValueWithMetadata valueWithMetadata =
          ValueWithMetadata.error(
              ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
              eventsAndPostables.first,
              eventsAndPostables.second);
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

  abstract <T extends SkyValue> EvaluationResult<T> constructResultExceptionally(
      Iterable<SkyKey> skyKeys,
      @Nullable Map<SkyKey, ValueWithMetadata> bubbleErrorInfo,
      boolean catastrophe)
      throws InterruptedException, E;

  /**
   * Constructs an {@link EvaluationResult} from the {@link #graph}. Looks for cycles if there are
   * unfinished nodes but no error was already found through bubbling up (as indicated by {@code
   * bubbleErrorInfo} being null).
   *
   * <p>{@code visitor} may be null, but only in the case where all graph entries corresponding to
   * {@code skyKeys} are known to be in the DONE state ({@code entry.isDone()} returns true).
   */
  <T extends SkyValue> EvaluationResult<T> constructResult(
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
    boolean nonCycleErrorFound = false;
    for (SkyKey skyKey : skyKeys) {
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
        nonCycleErrorFound = true;
        result.addError(skyKey, errorInfo);
        continue;
      }
      if (value == null) {
        // Note that we must be in the keepGoing case. Only make this value an error if it doesn't
        // have a value. The error shouldn't matter to the caller since the value succeeded after a
        // fashion.
        nonCycleErrorFound = true;
        result.addError(skyKey, errorInfo);
      } else {
        result.addResult(skyKey, value);
      }
    }
    if (!cycleRoots.isEmpty()) {
      cycleDetector.checkForCycles(cycleRoots, result, evaluatorContext);
    }
    if (catastrophe && bubbleErrorInfo != null && !result.hasCatastrophe()) {
      // We may not have a top-level node completed. Inform the caller of at least one catastrophic
      // exception that shut down the evaluation so that it has some context.
      // TODO(b/159006108): Sometimes we get here and not every exception is catastrophic, so we
      //  alert when that happens. If we didn't need to guard against that case, we could simply
      //  take the last element of bubbleErrorInfo#values() and make that the catastrophe.
      boolean catastropheFound = false;
      @Nullable Exception nonCatastrophicExceptionForBugHandler = null;
      for (ValueWithMetadata valueWithMetadata : bubbleErrorInfo.values()) {
        ErrorInfo errorInfo =
            Preconditions.checkNotNull(
                valueWithMetadata.getErrorInfo(),
                "bubbleErrorInfo should have contained element with errorInfo: %s",
                bubbleErrorInfo);
        if (errorInfo.isCatastrophic()) {
          if (!result.hasCatastrophe()) {
            result.setCatastrophe(errorInfo.getException());
          }
          catastropheFound = true;
        } else {
          // Alert for the known bug of a non-catastrophic exception.
          BugReport.sendBugReport(
              new IllegalStateException(
                  String.format(
                      "bubbleErrorInfo should have contained element with catastrophe: %s"
                          + " (bubbleErrorInfo: %s)",
                      valueWithMetadata, bubbleErrorInfo)));
          if (errorInfo.getException() != null) {
            nonCatastrophicExceptionForBugHandler = errorInfo.getException();
          }
        }
      }
      if (!catastropheFound && !nonCycleErrorFound) {
        Preconditions.checkNotNull(
            nonCatastrophicExceptionForBugHandler,
            "There were no exceptions in bubbleErrorInfo despite a catastrophic failure (%s)",
            bubbleErrorInfo);
        // Alert for the never-seen bug of *no* catastrophic exceptions.
        BugReport.sendBugReport(
            new IllegalStateException(
                "No element in bubbleErrorInfo was catastrophic: " + bubbleErrorInfo,
                nonCatastrophicExceptionForBugHandler));
        result.setCatastrophe(nonCatastrophicExceptionForBugHandler);
      }
    }
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
  static SkyValue maybeGetValueFromError(
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
      Map<SkyKey, SkyValue> injectionMap,
      Version version,
      EvaluableGraph graph,
      DirtyTrackingProgressReceiver progressReceiver)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> prevNodeEntries =
        graph.createIfAbsentBatch(null, Reason.OTHER, injectionMap.keySet());
    for (Map.Entry<SkyKey, SkyValue> injectionEntry : injectionMap.entrySet()) {
      SkyKey key = injectionEntry.getKey();
      SkyValue value = injectionEntry.getValue();
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
      prevEntry.setValue(value, version);
      // Now that this key's injected value is set, it is no longer dirty.
      progressReceiver.injected(key);
    }
  }
}
