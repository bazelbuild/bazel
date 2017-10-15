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

import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
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
 * <p>This class is not intended for direct use, and is only exposed as public for use in evaluation
 * implementations outside of this package.
 */
public class ParallelEvaluator extends AbstractParallelEvaluator implements Evaluator {

  private final CycleDetector cycleDetector;

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      int threadCount,
      DirtyTrackingProgressReceiver progressReceiver) {
    super(
        graph,
        graphVersion,
        skyFunctions,
        reporter,
        emittedEventState,
        storedEventFilter,
        errorInfoManager,
        keepGoing,
        threadCount,
        progressReceiver);
    cycleDetector = new SimpleCycleDetector();
  }

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      ForkJoinPool forkJoinPool,
      CycleDetector cycleDetector) {
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
        forkJoinPool);
    this.cycleDetector = cycleDetector;
  }

  private void informProgressReceiverThatValueIsDone(SkyKey key, NodeEntry entry)
      throws InterruptedException {
    if (evaluatorContext.getProgressReceiver() != null) {
      Preconditions.checkState(entry.isDone(), entry);
      SkyValue value = entry.getValue();
      Version valueVersion = entry.getVersion();
      Preconditions.checkState(
          valueVersion.atMost(evaluatorContext.getGraphVersion()),
          "%s should be at most %s in the version partial ordering",
          valueVersion,
          evaluatorContext.getGraphVersion());
      // For most nodes we do not inform the progress receiver if they were already done when we
      // retrieve them, but top-level nodes are presumably of more interest.
      // If valueVersion is not equal to graphVersion, it must be less than it (by the
      // Preconditions check above), and so the node is clean.
      evaluatorContext
          .getProgressReceiver()
          .evaluated(
              key,
              Suppliers.ofInstance(value),
              valueVersion.equals(evaluatorContext.getGraphVersion())
                  ? EvaluationState.BUILT
                  : EvaluationState.CLEAN);
    }
  }

  @Override
  @ThreadCompatible
  public <T extends SkyValue> EvaluationResult<T> eval(Iterable<? extends SkyKey> skyKeys)
      throws InterruptedException {
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
      // sanity checking).
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
        // sanity checking).
        return constructResult(cachedErrorKeys, null, /*catastrophe=*/ false);
      }
    }

    // We delay this check until we know that some kind of evaluation is necessary, since !keepGoing
    // and !keepsEdges are incompatible only in the case of a failed evaluation -- there is no
    // need to be overly harsh to callers who are just trying to retrieve a cached result.
    Preconditions.checkState(
        evaluatorContext.keepGoing()
            || !(graph instanceof InMemoryGraphImpl)
            || ((InMemoryGraphImpl) graph).keepsEdges(),
        "nokeep_going evaluations are not allowed if graph edges are not kept: %s",
        skyKeys);

    Profiler.instance().startTask(ProfilerTask.SKYFRAME_EVAL, skyKeySet);
    try {
      return doMutatingEvaluation(skyKeySet);
    } finally {
      Profiler.instance().completeTask(ProfilerTask.SKYFRAME_EVAL);
    }
  }

  @ThreadCompatible
  private <T extends SkyValue> EvaluationResult<T> doMutatingEvaluation(
      ImmutableSet<SkyKey> skyKeys) throws InterruptedException {
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
    for (Entry<SkyKey, ? extends NodeEntry> e :
        graph.createIfAbsentBatch(null, Reason.PRE_OR_POST_EVALUATION, skyKeys).entrySet()) {
      SkyKey skyKey = e.getKey();
      NodeEntry entry = e.getValue();
      // This must be equivalent to the code in enqueueChild above, in order to be thread-safe.
      switch (entry.addReverseDepAndCheckIfDone(null)) {
        case NEEDS_SCHEDULING:
          evaluatorContext.getVisitor().enqueueEvaluation(skyKey);
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
    return waitForCompletionAndConstructResult(skyKeys);
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
      if (!evaluatorContext.keepGoing()) {
        bubbleErrorInfo = bubbleErrorUp(errorInfo, errorKey, skyKeys);
      } else {
        Preconditions.checkState(
            errorInfo.isCatastrophic(),
            "Scheduler exception only thrown for catastrophe in keep_going evaluation: %s",
            e);
        catastrophe = true;
        // Bubbling the error up requires that graph edges are present for done nodes. This is not
        // always the case in a keepGoing evaluation, since it is assumed that done nodes do not
        // need to be traversed. In this case, we hope the caller is tolerant of a possibly empty
        // result, and return prematurely.
        bubbleErrorInfo =
            ImmutableMap.of(
                errorKey,
                ValueWithMetadata.wrapWithMetadata(
                    graph.get(null, Reason.ERROR_BUBBLING, errorKey).getValueMaybeWithMetadata()));
      }
    }
    Preconditions.checkState(
        evaluatorContext.getVisitor().getCrashes().isEmpty(),
        evaluatorContext.getVisitor().getCrashes());

    // Successful evaluation, either because keepGoing or because we actually did succeed.
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
   */
  private Map<SkyKey, ValueWithMetadata> bubbleErrorUp(
      final ErrorInfo leafFailure, SkyKey errorKey, Iterable<SkyKey> skyKeys)
      throws InterruptedException {
    Set<SkyKey> rootValues = ImmutableSet.copyOf(skyKeys);
    ErrorInfo error = leafFailure;
    Map<SkyKey, ValueWithMetadata> bubbleErrorInfo = new HashMap<>();
    boolean externalInterrupt = false;
    while (true) {
      NodeEntry errorEntry = Preconditions.checkNotNull(
          graph.get(null, Reason.ERROR_BUBBLING, errorKey),
          errorKey);
      Iterable<SkyKey> reverseDeps =
          errorEntry.isDone()
              ? errorEntry.getReverseDepsForDoneEntry()
              : errorEntry.getInProgressReverseDeps();
      // We should break from loop only when node doesn't have any parents.
      if (Iterables.isEmpty(reverseDeps)) {
        Preconditions.checkState(rootValues.contains(errorKey),
            "Current key %s has to be a top-level key: %s", errorKey, rootValues);
        break;
      }
      SkyKey parent = null;
      NodeEntry parentEntry = null;
      for (SkyKey bubbleParent : reverseDeps) {
        if (bubbleErrorInfo.containsKey(bubbleParent)) {
          // We are in a cycle. Don't try to bubble anything up -- cycle detection will kick in.
          return null;
        }
        NodeEntry bubbleParentEntry = Preconditions.checkNotNull(
            graph.get(errorKey, Reason.ERROR_BUBBLING, bubbleParent),
            "parent %s of %s not in graph",
            bubbleParent,
            errorKey);
        // Might be the parent that requested the error.
        if (bubbleParentEntry.isDone()) {
          // This parent is cached from a previous evaluate call. We shouldn't bubble up to it
          // since any error message produced won't be meaningful to this evaluate call.
          // The child error must also be cached from a previous build.
          Preconditions.checkState(errorEntry.isDone(), "%s %s", errorEntry, bubbleParentEntry);
          Version parentVersion = bubbleParentEntry.getVersion();
          Version childVersion = errorEntry.getVersion();
          Preconditions.checkState(
              childVersion.atMost(evaluatorContext.getGraphVersion())
                  && !childVersion.equals(evaluatorContext.getGraphVersion()),
              "child entry is not older than the current graph version, but had a done parent. "
                  + "child: %s childEntry: %s, childVersion: %s"
                  + "bubbleParent: %s bubbleParentEntry: %s, parentVersion: %s, graphVersion: %s",
              errorKey,
              errorEntry,
              childVersion,
              bubbleParent,
              bubbleParentEntry,
              parentVersion,
              evaluatorContext.getGraphVersion());
          Preconditions.checkState(
              parentVersion.atMost(evaluatorContext.getGraphVersion())
                  && !parentVersion.equals(evaluatorContext.getGraphVersion()),
              "parent entry is not older than the current graph version. "
                  + "child: %s childEntry: %s, childVersion: %s"
                  + "bubbleParent: %s bubbleParentEntry: %s, parentVersion: %s, graphVersion: %s",
              errorKey,
              errorEntry,
              childVersion,
              bubbleParent,
              bubbleParentEntry,
              parentVersion,
              evaluatorContext.getGraphVersion());
          continue;
        }
        if (evaluatorContext.getProgressReceiver().isInflight(bubbleParent)
            && bubbleParentEntry.getTemporaryDirectDeps().expensiveContains(errorKey)) {
          // Only bubble up to parent if it's part of this build. If this node was dirtied and
          // re-evaluated, but in a build without this parent, we may try to bubble up to that
          // parent. Don't -- it's not part of the build.
          // Similarly, the parent may not yet have requested this dep in its dirtiness-checking
          // process. Don't bubble up to it in that case either.
          parent = bubbleParent;
          parentEntry = bubbleParentEntry;
          break;
        }
      }
      if (parent == null) {
        Preconditions.checkState(
            rootValues.contains(errorKey),
            "Current key %s has to be a top-level key: %s, %s",
            errorKey,
            rootValues,
            errorEntry);
        break;
      }
      Preconditions.checkNotNull(parentEntry, "%s %s", errorKey, parent);
      errorKey = parent;
      SkyFunction factory = evaluatorContext.getSkyFunctions().get(parent.functionName());
      if (parentEntry.isDirty()) {
        switch (parentEntry.getDirtyState()) {
          case CHECK_DEPENDENCIES:
            // If this value's child was bubbled up to, it did not signal this value, and so we must
            // manually make it ready to build.
            parentEntry.signalDep();
            // Fall through to NEEDS_REBUILDING, since state is now NEEDS_REBUILDING.
          case NEEDS_REBUILDING:
            maybeMarkRebuilding(parentEntry);
            // Fall through to REBUILDING.
          case REBUILDING:
            break;
          default:
            throw new AssertionError(parent + " not in valid dirty state: " + parentEntry);
        }
      }
      SkyFunctionEnvironment env =
          new SkyFunctionEnvironment(
              parent,
              new GroupedList<SkyKey>(),
              bubbleErrorInfo,
              ImmutableSet.<SkyKey>of(),
              evaluatorContext);
      externalInterrupt = externalInterrupt || Thread.currentThread().isInterrupted();
      try {
        // This build is only to check if the parent node can give us a better error. We don't
        // care about a return value.
        factory.compute(parent, env);
      } catch (InterruptedException interruptedException) {
        // Do nothing.
        // This throw happens if the builder requested the failed node, and then checked the
        // interrupted state later -- getValueOrThrow sets the interrupted bit after the failed
        // value is requested, to prevent the builder from doing too much work.
      } catch (SkyFunctionException builderException) {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
        ReifiedSkyFunctionException reifiedBuilderException =
            new ReifiedSkyFunctionException(builderException, parent);
        if (reifiedBuilderException.getRootCauseSkyKey().equals(parent)) {
          error = ErrorInfo.fromException(reifiedBuilderException,
              /*isTransitivelyTransient=*/ false);
          bubbleErrorInfo.put(
              errorKey,
              ValueWithMetadata.error(
                  ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
                  env.buildEvents(parentEntry, /*missingChildren=*/ true),
                  env.buildPosts(parentEntry)));
          continue;
        }
      } finally {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
      }
      // Builder didn't throw an exception, so just propagate this one up.
      bubbleErrorInfo.put(
          errorKey,
          ValueWithMetadata.error(
              ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
              env.buildEvents(parentEntry, /*missingChildren=*/ true),
              env.buildPosts(parentEntry)));
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
        catastrophe == (evaluatorContext.keepGoing() && bubbleErrorInfo != null),
        "Catastrophe not consistent with keepGoing mode and bubbleErrorInfo: %s %s %s %s",
        skyKeys,
        catastrophe,
        evaluatorContext.keepGoing(),
        bubbleErrorInfo);
    EvaluationResult.Builder<T> result = EvaluationResult.builder();
    List<SkyKey> cycleRoots = new ArrayList<>();
    for (SkyKey skyKey : skyKeys) {
      SkyValue unwrappedValue = maybeGetValueFromError(
          skyKey,
          graph.get(null, Reason.PRE_OR_POST_EVALUATION, skyKey),
          bubbleErrorInfo);
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
      evaluatorContext
          .getReplayingNestedSetPostableVisitor()
          .visit(valueWithMetadata.getTransitivePostables());
      // TODO(bazel-team): Verify that message replay is fast and works in failure
      // modes [skyframe-core]
      // Note that replaying events here is only necessary on null builds, because otherwise we
      // would have already printed the transitive messages after building these values.
      evaluatorContext
          .getReplayingNestedSetEventVisitor()
          .visit(valueWithMetadata.getTransitiveEvents());
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
      cycleDetector.checkForCycles(cycleRoots, result, evaluatorContext);
    }
    if (catastrophe) {
      // We may not have a top-level node completed. Inform the caller of the catastrophic exception
      // that shut down the evaluation so that it has some context.
      ErrorInfo errorInfo =
          Preconditions.checkNotNull(
              Iterables.getOnlyElement(bubbleErrorInfo.values()).getErrorInfo(),
              "bubbleErrorInfo should have contained element with errorInfo: %s",
              bubbleErrorInfo);
      Preconditions.checkState(
          errorInfo.isCatastrophic(),
          "bubbleErrorInfo should have contained element with catastrophe: %s",
          bubbleErrorInfo);
      result.setCatastrophe(errorInfo.getException());
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
        Preconditions.checkState(
            newState == DependencyState.NEEDS_SCHEDULING, "%s %s", key, prevEntry);
        // There was an existing entry for this key in the graph.
        // Get the node in the state where it is able to accept a value.

        // Check that the previous node has no dependencies. Overwriting a value with deps with an
        // injected value (which is by definition deps-free) needs a little additional bookkeeping
        // (removing reverse deps from the dependencies), but more importantly it's something that
        // we want to avoid, because it indicates confusion of input values and derived values.
        Preconditions.checkState(
            prevEntry.noDepsLastBuild(), "existing entry for %s has deps: %s", key, prevEntry);
        prevEntry.markRebuilding();
      }
      prevEntry.setValue(value, version);
      // Now that this key's injected value is set, it is no longer dirty.
      progressReceiver.injected(key);
    }
  }
}
