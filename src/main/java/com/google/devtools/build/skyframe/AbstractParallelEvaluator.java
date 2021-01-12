// Copyright 2017 The Bazel Authors. All rights reserved.
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

import static java.lang.Math.min;

import com.google.common.base.Preconditions;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.common.flogger.GoogleLogger;
import com.google.common.graph.ImmutableGraph;
import com.google.common.graph.Traverser;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.clock.BlazeClock;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.NodeState;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.ParallelEvaluatorContext.EnqueueParentBehavior;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunction.Restart;
import com.google.devtools.build.skyframe.SkyFunctionEnvironment.UndonePreviouslyRequestedDeps;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import com.google.devtools.build.skyframe.ThinNodeEntry.DirtyType;
import com.google.devtools.build.skyframe.proto.GraphInconsistency.Inconsistency;
import java.io.IOException;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Supplier;
import javax.annotation.Nullable;

/**
 * Defines the evaluation action used in the multi-threaded Skyframe evaluation, and constructs the
 * {@link ParallelEvaluatorContext} that the actions rely on.
 *
 * <p>This does not implement other parts of Skyframe evaluation setup and post-processing, such as
 * translating a set of requested top-level nodes into actions, or constructing an evaluation
 * result. Derived classes should do this.
 */
abstract class AbstractParallelEvaluator {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  final ProcessableGraph graph;
  final ParallelEvaluatorContext evaluatorContext;
  protected final CycleDetector cycleDetector;
  private final AtomicInteger globalEnqueuedIndex;

  AbstractParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      final ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      Supplier<ExecutorService> executorService,
      CycleDetector cycleDetector,
      EvaluationVersionBehavior evaluationVersionBehavior) {
    this.graph = graph;
    this.cycleDetector = cycleDetector;
    evaluatorContext =
        new ParallelEvaluatorContext(
            graph,
            graphVersion,
            skyFunctions,
            reporter,
            emittedEventState,
            keepGoing,
            progressReceiver,
            storedEventFilter,
            errorInfoManager,
            graphInconsistencyReceiver,
            () ->
                new NodeEntryVisitor(
                    AbstractQueueVisitor.createWithExecutorService(
                        executorService.get(),
                        /*failFastOnException=*/ true,
                        NodeEntryVisitor.NODE_ENTRY_VISITOR_ERROR_CLASSIFIER),
                    progressReceiver,
                    (skyKey, evaluationPriority) -> new Evaluate(evaluationPriority, skyKey)),
            evaluationVersionBehavior);
    this.globalEnqueuedIndex = new AtomicInteger();
  }

  /**
   * If the entry is dirty and not already rebuilding, puts it in a state so that it can rebuild.
   */
  static void maybeMarkRebuilding(NodeEntry entry) {
    if (entry.isDirty()
        && entry.getDirtyState() != DirtyState.REBUILDING
        && entry.getDirtyState() != DirtyState.FORCED_REBUILDING) {
      entry.markRebuilding();
    }
  }

  enum DirtyOutcome {
    ALREADY_PROCESSED,
    NEEDS_EVALUATION
  }

  /**
   * An action that evaluates a value.
   *
   * <p>{@link Comparable} for use in priority queues. Experimentally, grouping enqueued evaluations
   * together by parent leads to fewer in-flight evaluations and thus lower peak memory usage. Thus
   * we store the {@link #evaluationPriority} (coming from the {@link #globalEnqueuedIndex} and use
   * it for comparisons: later enqueuings should be evaluated earlier, to do a depth-first search,
   * except for re-enqueued nodes, which always get top priority.
   *
   * <p>This is not applicable when using a {@link ForkJoinPool}, since it does not allow for easy
   * work prioritization.
   */
  private class Evaluate implements ParallelEvaluatorContext.ComparableRunnable {
    private final int evaluationPriority;
    /** The name of the value to be evaluated. */
    private final SkyKey skyKey;

    private Evaluate(int evaluationPriority, SkyKey skyKey) {
      this.evaluationPriority = evaluationPriority;
      this.skyKey = skyKey;
    }

    @Override
    public int compareTo(ParallelEvaluatorContext.ComparableRunnable other) {
      // Put other one first, so larger values come first in priority queue.
      return Integer.compare(((Evaluate) other).evaluationPriority, this.evaluationPriority);
    }

    /**
     * Notes the rdep from the parent to the child, and then does the appropriate thing with the
     * child or the parent, returning whether the parent has both been signalled and also is ready
     * for evaluation (but wasn't enqueued).
     */
    private boolean enqueueChild(
        SkyKey skyKey,
        NodeEntry entry,
        SkyKey child,
        NodeEntry childEntry,
        boolean depAlreadyExists,
        int childEvaluationPriority,
        EnqueueParentBehavior enqueueParent)
        throws InterruptedException {
      Preconditions.checkState(!entry.isDone(), "%s %s", skyKey, entry);
      DependencyState dependencyState;
      try {
        dependencyState =
            depAlreadyExists
                ? childEntry.checkIfDoneForDirtyReverseDep(skyKey)
                : childEntry.addReverseDepAndCheckIfDone(skyKey);
      } catch (IllegalStateException e) {
        // Add some more context regarding crashes.
        throw new IllegalStateException("child key: " + child + " error: " + e.getMessage(), e);
      }
      switch (dependencyState) {
        case DONE:
          switch (enqueueParent) {
            case SIGNAL:
              return entry.signalDep(childEntry.getVersion(), child);
            case ENQUEUE:
              if (entry.signalDep(childEntry.getVersion(), child)) {
                // Maximum priority, since this node has already started evaluation before, and we
                // want it off our plate.
                evaluatorContext.getVisitor().enqueueEvaluation(skyKey, Integer.MAX_VALUE);
              }
              break;
            case NO_ACTION:
              break;
          }
          break;
        case ALREADY_EVALUATING:
          break;
        case NEEDS_SCHEDULING:
          evaluatorContext.getVisitor().enqueueEvaluation(child, childEvaluationPriority);
          break;
      }
      return false;
    }

    /**
     * Returns true if this depGroup consists of the error transience value and the error transience
     * value is newer than the entry, meaning that the entry must be re-evaluated.
     */
    private boolean invalidatedByErrorTransience(Collection<SkyKey> depGroup, NodeEntry entry)
        throws InterruptedException {
      return depGroup.size() == 1
          && depGroup.contains(ErrorTransienceValue.KEY)
          && !graph
              .get(null, Reason.OTHER, ErrorTransienceValue.KEY)
              .getVersion()
              .atMost(entry.getVersion());
    }

    private DirtyOutcome maybeHandleDirtyNode(NodeEntry state) throws InterruptedException {
      while (state.getDirtyState().equals(DirtyState.CHECK_DEPENDENCIES)) {
        // Evaluating a dirty node for the first time, and checking its children to see if any
        // of them have changed. Note that there must be dirty children for this to happen.

        // Check the children group by group -- we don't want to evaluate a value that is no
        // longer needed because an earlier dependency changed. For example, //foo:foo depends
        // on target //bar:bar and is built. Then foo/BUILD is modified to remove the dependence
        // on bar, and bar/BUILD is deleted. Reloading //bar:bar would incorrectly throw an
        // exception. To avoid this, we must reload foo/BUILD first, at which point we will
        // discover that it has changed, and re-evaluate target //foo:foo from scratch.
        // On the other hand, when an action requests all of its inputs, we can safely check all
        // of them in parallel on a subsequent build. So we allow checking an entire group in
        // parallel here, if the node builder requested a group last build.
        // Note: every dep returned here must either have this node re-registered for it (using
        // checkIfDoneForDirtyReverseDep) and be registered as a direct dep of this node, or have
        // its reverse dep on this node removed. Failing to do either one of these would result in
        // a graph inconsistency, where the child had a reverse dep on this node, but this node
        // had no kind of dependency on the child.
        List<SkyKey> directDepsToCheck = state.getNextDirtyDirectDeps();

        if (invalidatedByErrorTransience(directDepsToCheck, state)) {
          // If this dep is the ErrorTransienceValue and the ErrorTransienceValue has been
          // updated then we need to force a rebuild. We would like to just signal the entry as
          // usual, but we can't, because then the ErrorTransienceValue would remain as a dep,
          // which would be incorrect if, for instance, the value re-evaluated to a non-error.
          state.forceRebuild();
          graph.get(skyKey, Reason.RDEP_REMOVAL, ErrorTransienceValue.KEY).removeReverseDep(skyKey);
          return DirtyOutcome.NEEDS_EVALUATION;
        }
        Map<SkyKey, ? extends NodeEntry> entriesToCheck = null;
        if (!evaluatorContext.keepGoing()) {
          // This check ensures that we maintain the invariant that if a node with an error is
          // reached during a no-keep-going build, none of its currently building parents
          // finishes building. If the child isn't done building yet, it will detect on its own
          // that it has an error (see the VERIFIED_CLEAN case below). On the other hand, if it
          // is done, then it is the parent's responsibility to notice that, which we do here.
          // We check the deps for errors so that we don't continue building this node if it has
          // a child error.
          entriesToCheck = graph.getBatch(skyKey, Reason.OTHER, directDepsToCheck);
          for (Map.Entry<SkyKey, ? extends NodeEntry> entry : entriesToCheck.entrySet()) {
            NodeEntry nodeEntryToCheck = entry.getValue();
            SkyValue valueMaybeWithMetadata = nodeEntryToCheck.getValueMaybeWithMetadata();
            if (valueMaybeWithMetadata == null) {
              continue;
            }
            ErrorInfo maybeErrorInfo = ValueWithMetadata.getMaybeErrorInfo(valueMaybeWithMetadata);
            if (maybeErrorInfo == null) {
              continue;
            }
            // This child has an error. We add a dep from this node to it and throw an exception
            // coming from it.
            SkyKey errorKey = entry.getKey();
            state.addTemporaryDirectDeps(GroupedListHelper.create(errorKey));
            nodeEntryToCheck.checkIfDoneForDirtyReverseDep(skyKey);
            // Perform the necessary bookkeeping for any deps that are not being used.
            for (Map.Entry<SkyKey, ? extends NodeEntry> depEntry : entriesToCheck.entrySet()) {
              if (!depEntry.getKey().equals(errorKey)) {
                depEntry.getValue().removeReverseDep(skyKey);
              }
            }
            if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
              // An error was already thrown in the evaluator. Don't do anything here.
              return DirtyOutcome.ALREADY_PROCESSED;
            }
            throw SchedulerException.ofError(maybeErrorInfo, errorKey, ImmutableSet.of(skyKey));
          }
        }
        // It is safe to add these deps back to the node -- even if one of them has changed, the
        // contract of pruning is that the node will request these deps again when it rebuilds.
        // We must add these deps before enqueuing them, so that the node knows that it depends
        // on them. If one of these deps is the error transience node, the check we did above
        // in #invalidatedByErrorTransience means that the error transience node is not newer
        // than this node, so we are going to mark it clean (since the error transience node is
        // always the last dep).
        state.addTemporaryDirectDepsGroupToDirtyEntry(directDepsToCheck);
        DepsReport depsReport = graph.analyzeDepsDoneness(skyKey, directDepsToCheck);
        Collection<SkyKey> unknownStatusDeps =
            depsReport.hasInformation() ? depsReport : directDepsToCheck;
        boolean needsScheduling = false;
        for (int i = 0; i < directDepsToCheck.size() - unknownStatusDeps.size(); i++) {
          // Since all of these nodes were done at an earlier version than this one, we may safely
          // signal with the minimal version, since they cannot trigger a re-evaluation.
          needsScheduling = state.signalDep(MinimalVersion.INSTANCE, /*childForDebugging=*/ null);
        }
        if (needsScheduling) {
          Preconditions.checkState(
              unknownStatusDeps.isEmpty(),
              "Ready without all deps checked? %s %s %s",
              skyKey,
              state,
              unknownStatusDeps);
          continue;
        }
        if (entriesToCheck == null || depsReport.hasInformation()) {
          entriesToCheck = graph.getBatch(skyKey, Reason.ENQUEUING_CHILD, unknownStatusDeps);
        }
        boolean parentIsSignalledAndReady =
            handleKnownChildrenForDirtyNode(
                unknownStatusDeps,
                entriesToCheck,
                state,
                globalEnqueuedIndex.incrementAndGet(),
                EnqueueParentBehavior.SIGNAL);
        if (!parentIsSignalledAndReady
            || evaluatorContext.getVisitor().shouldPreventNewEvaluations()) {
          return DirtyOutcome.ALREADY_PROCESSED;
        }
        // If we're here, then we may proceed to the rest of the method and continue processing
        // the node intra-thread. This is a performance optimization: By not enqueuing the node,
        // we avoid contention on the queue data structure (between concurrent threads
        // enqueueing and dequeueing), and we also save wall time since the node gets processed
        // now rather than at some point in the future.
      }
      switch (state.getDirtyState()) {
        case VERIFIED_CLEAN:
          // No child has a changed value. This node can be marked done and its parents signaled
          // without any re-evaluation.
          NodeEntry.NodeValueAndRdepsToSignal nodeValueAndRdeps = state.markClean();
          Set<SkyKey> rDepsToSignal = nodeValueAndRdeps.getRdepsToSignal();
          // Make sure to replay events once change-pruned
          replay(ValueWithMetadata.wrapWithMetadata(nodeValueAndRdeps.getValue()));
          // Tell the receiver that the value was not actually changed this run.
          evaluatorContext
              .getProgressReceiver()
              .evaluated(
                  skyKey, null, new EvaluationSuccessStateSupplier(state), EvaluationState.CLEAN);
          if (!evaluatorContext.keepGoing() && state.getErrorInfo() != null) {
            if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
              return DirtyOutcome.ALREADY_PROCESSED;
            }
            throw SchedulerException.ofError(state.getErrorInfo(), skyKey, rDepsToSignal);
          }
          evaluatorContext.signalValuesAndEnqueueIfReady(
              skyKey, rDepsToSignal, state.getVersion(), EnqueueParentBehavior.ENQUEUE);
          return DirtyOutcome.ALREADY_PROCESSED;
        case NEEDS_REBUILDING:
          state.markRebuilding();
          return DirtyOutcome.NEEDS_EVALUATION;
        case NEEDS_FORCED_REBUILDING:
          state.forceRebuild();
          return DirtyOutcome.NEEDS_EVALUATION;
        case REBUILDING:
        case FORCED_REBUILDING:
          return DirtyOutcome.NEEDS_EVALUATION;
        default:
          throw new IllegalStateException("key: " + skyKey + ", entry: " + state);
      }
    }

    /**
     * Returns whether the parent has both been signalled and also is ready for evaluation (but
     * wasn't enqueued).
     */
    private boolean handleKnownChildrenForDirtyNode(
        Collection<SkyKey> knownChildren,
        Map<SkyKey, ? extends NodeEntry> oldChildren,
        NodeEntry state,
        int childEvaluationPriority,
        EnqueueParentBehavior enqueueParent)
        throws InterruptedException {
      boolean parentIsSignalledAndReady = false;
      if (oldChildren.size() != knownChildren.size()) {
        GraphInconsistencyReceiver inconsistencyReceiver =
            evaluatorContext.getGraphInconsistencyReceiver();
        Set<SkyKey> missingChildren =
            Sets.difference(ImmutableSet.copyOf(knownChildren), oldChildren.keySet());
        if (!missingChildren.isEmpty()) {
          inconsistencyReceiver.noteInconsistencyAndMaybeThrow(
              skyKey, missingChildren, Inconsistency.DIRTY_PARENT_HAD_MISSING_CHILD);
        }
        Map<SkyKey, ? extends NodeEntry> recreatedEntries =
            graph.createIfAbsentBatch(skyKey, Reason.ENQUEUING_CHILD, missingChildren);
        for (Map.Entry<SkyKey, ? extends NodeEntry> recreatedEntry : recreatedEntries.entrySet()) {
          parentIsSignalledAndReady |=
              enqueueChild(
                  skyKey,
                  state,
                  recreatedEntry.getKey(),
                  recreatedEntry.getValue(),
                  /*depAlreadyExists=*/ false,
                  childEvaluationPriority,
                  enqueueParent);
        }
      }
      for (Map.Entry<SkyKey, ? extends NodeEntry> e : oldChildren.entrySet()) {
        SkyKey directDep = e.getKey();
        NodeEntry directDepEntry = e.getValue();
        parentIsSignalledAndReady |=
            enqueueChild(
                skyKey,
                state,
                directDep,
                directDepEntry,
                /*depAlreadyExists=*/ true,
                childEvaluationPriority,
                enqueueParent);
      }
      return parentIsSignalledAndReady;
    }

    @Override
    public void run() {
      SkyFunctionEnvironment env = null;
      try {
        NodeEntry state =
            Preconditions.checkNotNull(graph.get(null, Reason.EVALUATION, skyKey), skyKey);
        Preconditions.checkState(state.isReady(), "%s %s", skyKey, state);
        try {
          evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.CHECK_DIRTY);
          if (maybeHandleDirtyNode(state) == DirtyOutcome.ALREADY_PROCESSED) {
            return;
          }
        } finally {
          evaluatorContext.getProgressReceiver().stateEnding(skyKey, NodeState.CHECK_DIRTY, -1);
        }

        ImmutableSet<SkyKey> oldDeps = state.getAllRemainingDirtyDirectDeps();
        try {
          evaluatorContext
              .getProgressReceiver()
              .stateStarting(skyKey, NodeState.INITIALIZING_ENVIRONMENT);
          env =
              new SkyFunctionEnvironment(
                  skyKey, state.getTemporaryDirectDeps(), oldDeps, evaluatorContext);
        } catch (UndonePreviouslyRequestedDeps undonePreviouslyRequestedDeps) {
          // If a previously requested dep is no longer done, restart this node from scratch.
          restart(skyKey, state);
          // Top priority since this node has already been evaluating, so get it off our plate.
          evaluatorContext.getVisitor().enqueueEvaluation(skyKey, Integer.MAX_VALUE);
          return;
        } finally {
          evaluatorContext
              .getProgressReceiver()
              .stateEnding(skyKey, NodeState.INITIALIZING_ENVIRONMENT, -1);
        }
        SkyFunctionName functionName = skyKey.functionName();
        SkyFunction factory =
            Preconditions.checkNotNull(
                evaluatorContext.getSkyFunctions().get(functionName),
                "Unable to find SkyFunction '%s' for node with key %s, %s",
                functionName,
                skyKey,
                state);

        SkyValue value = null;
        long startTimeNanos = BlazeClock.instance().nanoTime();
        try {
          try {
            evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.COMPUTE);
            value = factory.compute(skyKey, env);
          } finally {
            long elapsedTimeNanos = BlazeClock.instance().nanoTime() - startTimeNanos;
            evaluatorContext
                .getProgressReceiver()
                .stateEnding(skyKey, NodeState.COMPUTE, elapsedTimeNanos);
            if (elapsedTimeNanos > 0) {
              Profiler.instance()
                  .logSimpleTaskDuration(
                      startTimeNanos,
                      Duration.ofNanos(elapsedTimeNanos),
                      ProfilerTask.SKYFUNCTION,
                      skyKey.functionName().getName());
            }
          }
        } catch (final SkyFunctionException builderException) {
          ReifiedSkyFunctionException reifiedBuilderException =
              new ReifiedSkyFunctionException(builderException);
          // In keep-going mode, we do not let SkyFunctions complete with a thrown error if they
          // have missing deps. Instead, we wait until their deps are done and restart the
          // SkyFunction, so we can have a definitive error and definitive graph structure, thus
          // avoiding non-determinism. It's completely reasonable for SkyFunctions to throw eagerly
          // because they do not know if they are in keep-going mode.
          if (!evaluatorContext.keepGoing() || !env.valuesMissing()) {
            boolean shouldFailFast =
                !evaluatorContext.keepGoing() || builderException.isCatastrophic();
            if (shouldFailFast) {
              // After we commit this error to the graph but before the doMutatingEvaluation call
              // completes with the error there is a race-like opportunity for the error to be used,
              // either by an in-flight computation or by a future computation.
              if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
                // This is not the first error encountered, so we ignore it so that we can terminate
                // with the first error.
                return;
              } else {
                logger.atWarning().withCause(builderException).log(
                    "Aborting evaluation while evaluating %s", skyKey);
              }
            }

            if (maybeHandleRegisteringNewlyDiscoveredDepsForDoneEntry(
                skyKey, state, oldDeps, env, evaluatorContext.keepGoing())) {
              // A newly requested dep transitioned from done to dirty before this node finished.
              // If shouldFailFast is true, this node won't be signalled by any such newly dirtied
              // dep (because new evaluations have been prevented), and this node is responsible for
              // throwing the SchedulerException below.
              // Otherwise, this node will be signalled again, and so we should return.
              if (!shouldFailFast) {
                return;
              }
            }
            boolean isTransitivelyTransient =
                reifiedBuilderException.isTransient()
                    || env.isAnyDirectDepErrorTransitivelyTransient()
                    || env.isAnyNewlyRequestedDepErrorTransitivelyTransient();
            ErrorInfo errorInfo =
                evaluatorContext
                    .getErrorInfoManager()
                    .fromException(skyKey, reifiedBuilderException, isTransitivelyTransient);
            // TODO(b/166268889): Remove when resolved. ActionExecutionValues are ending up with
            //  IOExceptions in them.
            if (isTransitivelyTransient
                && !shouldFailFast
                && errorInfo.getException() instanceof IOException) {
              // This is essentially unconditionally logged, and not often. Ok to evaluate eagerly.
              String keyString = skyKey.toString();
              String errorString = errorInfo.toString();
              logger.atInfo().log(
                  "Got IOException for %s (%s)",
                  keyString.substring(0, min(1000, keyString.length())),
                  errorString.substring(0, min(1000, errorString.length())));
            }
            env.setError(state, errorInfo);
            Set<SkyKey> rdepsToBubbleUpTo =
                env.commit(
                    state,
                    shouldFailFast ? EnqueueParentBehavior.SIGNAL : EnqueueParentBehavior.ENQUEUE);
            if (!shouldFailFast) {
              return;
            }
            throw SchedulerException.ofError(errorInfo, skyKey, rdepsToBubbleUpTo);
          }
        } catch (RuntimeException re) {
          // Programmer error (most likely NPE or a failed precondition in a SkyFunction). Output
          // some context together with the exception.
          String msg = prepareCrashMessage(skyKey, state.getInProgressReverseDeps());
          RuntimeException ex = new RuntimeException(msg, re);
          evaluatorContext.getVisitor().noteCrash(ex);
          throw ex;
        } finally {
          env.doneBuilding();
        }

        if (maybeHandleRestart(skyKey, state, value)) {
          cancelExternalDeps(env);
          // Top priority since this node has already been evaluating, so get it off our plate.
          evaluatorContext.getVisitor().enqueueEvaluation(skyKey, Integer.MAX_VALUE);
          return;
        }

        // Helper objects for all the newly requested deps that weren't known to the environment,
        // and may contain duplicate elements.
        GroupedListHelper<SkyKey> newDirectDeps = env.getNewlyRequestedDeps();

        if (value != null) {
          Preconditions.checkState(
              !env.valuesMissing(),
              "Evaluation of %s returned non-null value but requested dependencies that weren't "
                  + "computed yet (one of %s), NodeEntry: %s",
              skyKey,
              newDirectDeps,
              state);

          try {
            evaluatorContext.getProgressReceiver().stateStarting(skyKey, NodeState.COMMIT);
            if (maybeHandleRegisteringNewlyDiscoveredDepsForDoneEntry(
                skyKey, state, oldDeps, env, evaluatorContext.keepGoing())) {
              // A newly requested dep transitioned from done to dirty before this node finished.
              // This node will be signalled again, and so we should return.
              return;
            }
            env.setValue(value);
            env.commit(state, EnqueueParentBehavior.ENQUEUE);
          } finally {
            evaluatorContext.getProgressReceiver().stateEnding(skyKey, NodeState.COMMIT, -1);
          }
          return;
        }

        SkyKey childErrorKey = env.getDepErrorKey();
        if (childErrorKey != null) {
          Preconditions.checkState(
              !evaluatorContext.keepGoing(), "%s %s %s", skyKey, state, childErrorKey);
          // We encountered a child error in noKeepGoing mode, so we want to fail fast. But we first
          // need to add the edge between the current node and the child error it requested so that
          // error bubbling can occur. Note that this edge will subsequently be removed during graph
          // cleaning (since the current node will never be committed to the graph).
          NodeEntry childErrorEntry =
              Preconditions.checkNotNull(
                  graph.get(skyKey, Reason.OTHER, childErrorKey),
                  "skyKey: %s, state: %s childErrorKey: %s",
                  skyKey,
                  state,
                  childErrorKey);
          if (newDirectDeps.contains(childErrorKey)) {
            // Add this dep if it was just requested. In certain rare race conditions (see
            // MemoizingEvaluatorTest.cachedErrorCausesRestart) this dep may have already been
            // requested.
            state.addTemporaryDirectDeps(GroupedListHelper.create(childErrorKey));
            DependencyState childErrorState;
            if (oldDeps.contains(childErrorKey)) {
              childErrorState = childErrorEntry.checkIfDoneForDirtyReverseDep(skyKey);
            } else {
              childErrorState = childErrorEntry.addReverseDepAndCheckIfDone(skyKey);
            }
            if (childErrorState != DependencyState.DONE) {
              // The child in error may have transitioned from done to dirty between when this node
              // discovered the error and now. Notify the graph inconsistency receiver so that we
              // can crash if that's unexpected.
              // We don't enqueue the child, even if it returns NEEDS_SCHEDULING, because we are
              // about to shut down evaluation.
              evaluatorContext
                  .getGraphInconsistencyReceiver()
                  .noteInconsistencyAndMaybeThrow(
                      skyKey,
                      ImmutableList.of(childErrorKey),
                      Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
            }
          }
          SkyValue childErrorInfoMaybe =
              Preconditions.checkNotNull(
                  env.maybeGetValueFromErrorOrDeps(childErrorKey),
                  "dep error found but then lost while building: %s %s",
                  skyKey,
                  childErrorKey);
          ErrorInfo childErrorInfo =
              Preconditions.checkNotNull(
                  ValueWithMetadata.getMaybeErrorInfo(childErrorInfoMaybe),
                  "dep error found but then wasn't an error while building: %s %s %s",
                  skyKey,
                  childErrorKey,
                  childErrorInfoMaybe);
          evaluatorContext.getVisitor().preventNewEvaluations();
          throw SchedulerException.ofError(childErrorInfo, childErrorKey, ImmutableSet.of(skyKey));
        }

        // TODO(bazel-team): This code is not safe to interrupt, because we would lose the state in
        // newDirectDeps.

        // TODO(bazel-team): An ill-behaved SkyFunction can throw us into an infinite loop where we
        // add more dependencies on every run. [skyframe-core]

        // Add all the newly requested dependencies to the temporary direct deps. Note that
        // newDirectDeps does not contain any elements in common with the already existing temporary
        // direct deps. uniqueNewDeps will be the set of unique keys contained in newDirectDeps.
        Set<SkyKey> uniqueNewDeps = state.addTemporaryDirectDeps(newDirectDeps);

        List<ListenableFuture<?>> externalDeps = env.externalDeps;
        // If there were no newly requested dependencies, at least one of them was in error or there
        // is a bug in the SkyFunction implementation. The environment has collected its errors, so
        // we just order it to be built.
        if (uniqueNewDeps.isEmpty() && externalDeps == null) {
          // TODO(bazel-team): This means a bug in the SkyFunction. What to do?
          Preconditions.checkState(
              !env.getChildErrorInfos().isEmpty(),
              "Evaluation of SkyKey failed and no dependencies were requested: %s %s",
              skyKey,
              state);
          Preconditions.checkState(
              evaluatorContext.keepGoing(),
              "nokeep_going evaluation should have failed on first child error: %s %s %s",
              skyKey,
              state,
              env.getChildErrorInfos());
          // If the child error was catastrophic, committing this parent to the graph is not
          // necessary, but since we don't do error bubbling in catastrophes, it doesn't violate any
          // invariants either.
          env.commit(state, EnqueueParentBehavior.ENQUEUE);
          return;
        }

        // If there are external deps, we register that fact on the NodeEntry before we enqueue
        // child nodes in order to prevent the current node from being re-enqueued between here and
        // the call to registerExternalDeps below.
        if (externalDeps != null) {
          state.addExternalDep();
        }

        // We want to split apart the dependencies that existed for this node the last time we did
        // an evaluation and those that were introduced in this evaluation. To be clear, the prefix
        // "newDeps" refers to newly discovered this time around after a SkyFunction#compute call
        // and not to be confused with the oldDeps variable which refers to the last evaluation,
        // (ie) a prior call to ParallelEvaluator#eval).
        Set<SkyKey> newDepsThatWerentInTheLastEvaluation = Sets.difference(uniqueNewDeps, oldDeps);
        Set<SkyKey> newDepsThatWereInTheLastEvaluation =
            Sets.difference(uniqueNewDeps, newDepsThatWerentInTheLastEvaluation);

        int childEvaluationPriority = globalEnqueuedIndex.incrementAndGet();
        InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>>
            newDepsThatWerentInTheLastEvaluationNodes =
                graph.createIfAbsentBatchAsync(
                    skyKey, Reason.RDEP_ADDITION, newDepsThatWerentInTheLastEvaluation);
        handleKnownChildrenForDirtyNode(
            newDepsThatWereInTheLastEvaluation,
            graph.getBatch(skyKey, Reason.ENQUEUING_CHILD, newDepsThatWereInTheLastEvaluation),
            state,
            childEvaluationPriority,
            EnqueueParentBehavior.ENQUEUE);

        // Due to multi-threading, this can potentially cause the current node to be re-enqueued if
        // all 'new' children of this node are already done. Therefore, there should not be any
        // code after this loop, as it would potentially race with the re-evaluation in another
        // thread.
        for (Map.Entry<SkyKey, ? extends NodeEntry> e :
            newDepsThatWerentInTheLastEvaluationNodes.get().entrySet()) {
          SkyKey newDirectDep = e.getKey();
          NodeEntry newDirectDepEntry = e.getValue();
          enqueueChild(
              skyKey,
              state,
              newDirectDep,
              newDirectDepEntry,
              /*depAlreadyExists=*/ false,
              childEvaluationPriority,
              EnqueueParentBehavior.ENQUEUE);
        }
        if (externalDeps != null) {
          // This can cause the current node to be re-enqueued if all futures are already done.
          // This is an exception to the rule above that there must not be code below the for
          // loop. It is safe because we call state.addExternalDep above, which prevents
          // re-enqueueing of the current node in the above loop if externalDeps != null.
          evaluatorContext.getVisitor().registerExternalDeps(skyKey, state, externalDeps);
        }
        // Do not put any code here! Any code here can race with a re-evaluation of this same node
        // in another thread.
      } catch (InterruptedException ie) {
        // The current thread can be interrupted at various places during evaluation or while
        // committing the result in this method. Since we only register the future(s) with the
        // underlying AbstractQueueVisitor in the registerExternalDeps call above, we have to make
        // sure that any known futures are correctly canceled if we do not reach that call. Note
        // that it is safe to cancel a future multiple times.
        cancelExternalDeps(env);
        // InterruptedException cannot be thrown by Runnable.run, so we must wrap it.
        // Interrupts can be caught by both the Evaluator and the AbstractQueueVisitor.
        // The former will unwrap the IE and propagate it as is; the latter will throw a new IE.
        throw SchedulerException.ofInterruption(ie, skyKey);
      }
    }

    private void cancelExternalDeps(SkyFunctionEnvironment env) {
      if (env != null && env.externalDeps != null) {
        for (ListenableFuture<?> future : env.externalDeps) {
          future.cancel(/*mayInterruptIfRunning=*/ true);
        }
      }
    }

    private String prepareCrashMessage(SkyKey skyKey, Iterable<SkyKey> reverseDeps) {
      StringBuilder reverseDepDump = new StringBuilder();
      for (SkyKey key : reverseDeps) {
        if (reverseDepDump.length() > MAX_REVERSEDEP_DUMP_LENGTH) {
          reverseDepDump.append(", ...");
          break;
        }
        if (reverseDepDump.length() > 0) {
          reverseDepDump.append(", ");
        }
        reverseDepDump.append("'");
        reverseDepDump.append(key.toString());
        reverseDepDump.append("'");
      }

      return String.format(
          "Unrecoverable error while evaluating node '%s' (requested by nodes %s)",
          skyKey, reverseDepDump);
    }

    private static final int MAX_REVERSEDEP_DUMP_LENGTH = 1000;
  }

  protected void replay(ValueWithMetadata valueWithMetadata) {
    // Replaying actions is done on a small number of nodes, but potentially over a large dependency
    // graph. Under those conditions, using the regular NestedSet flattening with .toList() is more
    // efficient than using NestedSetVisitor's custom traversal logic.
    evaluatorContext
        .getReplayingNestedSetPostableVisitor()
        .visit(valueWithMetadata.getTransitivePostables().toList());
    evaluatorContext
        .getReplayingNestedSetEventVisitor()
        .visit(valueWithMetadata.getTransitiveEvents().toList());
  }

  /**
   * If {@code returnedValue} is a {@link Restart} value, then {@code entry} will be reset, and the
   * other nodes specified by {@code returnedValue.rewindGraph()} will be marked changed via
   * postorder DFS.
   *
   * <p>{@code returnedValue.rewindGraph()} must be empty or must contain {@code key}.
   *
   * <p>TODO(b/123993876): this should verify that edges in rewindGraph correspond to deps in the
   * Skyframe graph. Will require a safe way of requesting deps for nodes which may not be done.
   *
   * @return {@code returnedValue instanceof Restart}
   */
  // Nodes must be marked changed via postorder DFS. To see why, suppose we have this graph:
  //
  //   FailedNode   SomeOtherRdepOfR1
  //       |       /
  //       |  -----
  //       | /
  //       R1
  //       |
  //       R2
  //
  // Suppose FailedNode (FN) fails and requires that R1 and R2 must be dirtied and run again.
  // Suppose they aren't dirtied via postorder DFS, so R1 is dirtied first.
  //
  // Then, the evaluation thread working on dirtying these nodes is suspended.
  //
  // On a separate evaluation thread, SomeOtherRdepOfR1 requests R1. R1 is scheduled for evaluation,
  // checks its dep R2, and because R2 is done, R1 completes without scheduling R2 for evaluation.
  //
  // Then, the evaluation thread working on dirtying these nodes continues its work. It dirties
  // R2 and schedules FN for evaluation.
  //
  // When FN next evaluates, it requests R1, and because R1 is done, R2 is not scheduled for
  // evaluation, contrary to FN's expectations.
  private boolean maybeHandleRestart(SkyKey key, NodeEntry entry, SkyValue returnedValue)
      throws InterruptedException {
    if (!(returnedValue instanceof Restart)) {
      return false;
    }

    ImmutableGraph<SkyKey> rewindGraph = ((Restart) returnedValue).rewindGraph();
    if (rewindGraph.nodes().isEmpty()) {
      restart(key, entry);
      return true;
    }
    Preconditions.checkArgument(
        rewindGraph.nodes().contains(key),
        "rewindGraph must contain the key for the failed evaluation if it's not empty. key: %s, "
            + "rewindGraph: %s",
        key,
        rewindGraph);

    ImmutableList.Builder<SkyKey> builder = ImmutableList.builder();
    for (SkyKey k : Traverser.forGraph(rewindGraph).depthFirstPostOrder(key)) {
      if (!k.equals(key)) {
        builder.add(k);
      }
    }
    ImmutableList<SkyKey> additionalKeysToRestart = builder.build();
    if (!additionalKeysToRestart.isEmpty()) {
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              key, additionalKeysToRestart, Inconsistency.PARENT_FORCE_REBUILD_OF_CHILD);
    }

    Map<SkyKey, ? extends NodeEntry> additionalNodesToRestart =
        evaluatorContext.getBatchValues(key, Reason.INVALIDATION, additionalKeysToRestart);

    ArrayList<SkyKey> missingNodes = null;
    for (SkyKey keyToRestart : additionalKeysToRestart) {
      NodeEntry restartEntry = additionalNodesToRestart.get(keyToRestart);

      if (restartEntry == null) {
        if (missingNodes == null) {
          missingNodes = new ArrayList<>();
        }
        missingNodes.add(keyToRestart);
        continue;
      }

      // Nodes are marked "force-rebuild" to ensure that they run, and to allow them to evaluate to
      // a different value than before, even if their versions remain the same.
      if (restartEntry.markDirty(DirtyType.FORCE_REBUILD) != null) {
        evaluatorContext
            .getProgressReceiver()
            .invalidated(keyToRestart, EvaluationProgressReceiver.InvalidationState.DIRTY);
      }
    }

    if (missingNodes != null) {
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              key, missingNodes, Inconsistency.PARENT_FORCE_REBUILD_OF_MISSING_CHILD);
    }

    // TODO(b/19539699): rdeps of children have to be handled here. If the graph does not keep
    // edges, nothing has to be done, since there are no reverse deps to keep consistent. If the
    // graph keeps edges, it's a harder problem. The reverse deps could just be removed, but in the
    // case that this node is dirty, the deps shouldn't be removed, they should just be transformed
    // back to "known reverse deps" from "reverse deps declared during this evaluation" (the inverse
    // of NodeEntry#checkIfDoneForDirtyReverseDep). Such a method doesn't currently exist, but
    // could.
    restart(key, entry);
    return true;
  }

  private void restart(SkyKey key, NodeEntry entry) {
    evaluatorContext
        .getGraphInconsistencyReceiver()
        .noteInconsistencyAndMaybeThrow(key, /*otherKeys=*/ null, Inconsistency.RESET_REQUESTED);
    entry.resetForRestartFromScratch();
  }

  void propagateEvaluatorContextCrashIfAny() {
    if (!evaluatorContext.getVisitor().getCrashes().isEmpty()) {
      evaluatorContext
          .getReporter()
          .handle(Event.error("Crashes detected: " + evaluatorContext.getVisitor().getCrashes()));
      throw Preconditions.checkNotNull(
          Iterables.getFirst(evaluatorContext.getVisitor().getCrashes(), null));
    }
  }

  void propagateInterruption(SchedulerException e) throws InterruptedException {
    boolean mustThrowInterrupt = Thread.interrupted();
    Throwables.propagateIfPossible(e.getCause(), InterruptedException.class);
    if (mustThrowInterrupt) {
      // As per the contract of AbstractQueueVisitor#work, if an unchecked exception is thrown and
      // the build is interrupted, the thrown exception is what will be rethrown. Since the user
      // presumably wanted to interrupt the build, we ignore the thrown SchedulerException (which
      // doesn't indicate a programming bug) and throw an InterruptedException.
      throw new InterruptedException();
    }
  }

  /**
   * Add any newly discovered deps that were registered during the run of a SkyFunction that
   * finished by returning a value or throwing an error. SkyFunctions may throw errors even if all
   * their deps were not provided -- we trust that a SkyFunction might know it should throw an error
   * even if not all of its requested deps are done. However, that means we're assuming the
   * SkyFunction would throw that same error if all of its requested deps were done. Unfortunately,
   * there is no way to enforce that condition.
   *
   * <p>Returns {@code true} if any newly discovered dep is dirty when this node registers itself as
   * an rdep and if one of those dirty deps will schedule this node for evaluation.
   *
   * <p>This can happen if a newly discovered dep transitions from done to dirty between when this
   * node's evaluation accessed the dep's value and here. Adding this node as an rdep of that dep
   * (or checking that this node is an rdep of that dep) will cause this node to be signalled when
   * that dep completes.
   *
   * <p>If this returns {@code true}, this node should not actually finish, and this evaluation
   * attempt should make no changes to the node after this method returns, because a completing dep
   * may schedule a new evaluation attempt at any time.
   *
   * @throws InterruptedException
   */
  private boolean maybeHandleRegisteringNewlyDiscoveredDepsForDoneEntry(
      SkyKey skyKey,
      NodeEntry entry,
      ImmutableSet<SkyKey> oldDeps,
      SkyFunctionEnvironment env,
      boolean keepGoing)
      throws InterruptedException {
    Iterator<SkyKey> it = env.getNewlyRequestedDeps().iterator();
    if (!it.hasNext()) {
      return false;
    }

    // We don't expect any unfinished deps in a keep-going build.
    if (!keepGoing) {
      env.removeUndoneNewlyRequestedDeps();
    }

    Set<SkyKey> uniqueNewDeps = entry.addTemporaryDirectDeps(env.getNewlyRequestedDeps());
    Set<SkyKey> newlyAddedNewDeps = Sets.difference(uniqueNewDeps, oldDeps);
    Set<SkyKey> previouslyRegisteredNewDeps = Sets.difference(uniqueNewDeps, newlyAddedNewDeps);

    InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>> newlyAddedNewDepNodes =
        graph.getBatchAsync(skyKey, Reason.RDEP_ADDITION, newlyAddedNewDeps);

    // Dep entries in the following two loops may not be done, but they must be present. If the
    // graph permits an already declared child missing, we recreate the entry if necessary. In a
    // keep-going build, we normally expect all deps to be done. In a non-keep-going build, if
    // env.newlyRequestedDeps contained a key for a node that wasn't done, then it would have been
    // removed via removeUndoneNewlyRequestedDeps() just above this loop. However, with
    // intra-evaluation dirtying, a dep may not be done.
    boolean dirtyDepFound = false;
    boolean selfSignalled = false;

    Map<SkyKey, ? extends NodeEntry> previouslyRegisteredEntries =
        graph.getBatch(skyKey, Reason.SIGNAL_DEP, previouslyRegisteredNewDeps);
    for (SkyKey newDep : previouslyRegisteredNewDeps) {
      NodeEntry depEntry =
          getOrRecreateDepEntry(newDep, previouslyRegisteredEntries, skyKey, Reason.SIGNAL_DEP);
      DependencyState triState = depEntry.checkIfDoneForDirtyReverseDep(skyKey);
      switch (maybeHandleUndoneDepForDoneEntry(entry, depEntry, triState, skyKey, newDep)) {
        case DEP_DONE_SELF_SIGNALLED:
          selfSignalled = true;
          break;
        case DEP_DONE_SELF_NOT_SIGNALLED:
          break;
        case DEP_NOT_DONE:
          dirtyDepFound = true;
          break;
      }
    }

    for (SkyKey newDep : newlyAddedNewDeps) {
      NodeEntry depEntry =
          getOrRecreateDepEntry(newDep, newlyAddedNewDepNodes.get(), skyKey, Reason.RDEP_ADDITION);
      DependencyState triState = depEntry.addReverseDepAndCheckIfDone(skyKey);
      switch (maybeHandleUndoneDepForDoneEntry(entry, depEntry, triState, skyKey, newDep)) {
        case DEP_DONE_SELF_SIGNALLED:
          selfSignalled = true;
          break;
        case DEP_DONE_SELF_NOT_SIGNALLED:
          break;
        case DEP_NOT_DONE:
          dirtyDepFound = true;
          break;
      }
    }

    Preconditions.checkState(
        selfSignalled || dirtyDepFound || uniqueNewDeps.isEmpty(),
        "%s %s %s %s",
        skyKey,
        entry,
        newlyAddedNewDeps,
        previouslyRegisteredNewDeps);

    return !selfSignalled;
  }

  /**
   * Returns a {@link NodeEntry} for {@code depKey}.
   *
   * <p>If {@code depKey} is present in {@code depEntries}, its corresponding entry is returned.
   * Otherwise, if the evaluator permits {@link Inconsistency#ALREADY_DECLARED_CHILD_MISSING}, the
   * entry will be recreated.
   */
  private NodeEntry getOrRecreateDepEntry(
      SkyKey depKey, Map<SkyKey, ? extends NodeEntry> depEntries, SkyKey requestor, Reason reason)
      throws InterruptedException {
    NodeEntry depEntry = depEntries.get(depKey);
    if (depEntry == null) {
      List<SkyKey> missing = ImmutableList.of(depKey);
      evaluatorContext
          .getGraphInconsistencyReceiver()
          .noteInconsistencyAndMaybeThrow(
              depKey, missing, Inconsistency.ALREADY_DECLARED_CHILD_MISSING);
      depEntry =
          Preconditions.checkNotNull(
              graph.createIfAbsentBatch(requestor, reason, missing).get(depKey), depKey);
    }
    return depEntry;
  }

  private enum MaybeHandleUndoneDepResult {
    DEP_DONE_SELF_SIGNALLED,
    DEP_DONE_SELF_NOT_SIGNALLED,
    DEP_NOT_DONE
  }

  /**
   * Returns {@link MaybeHandleUndoneDepResult#DEP_NOT_DONE} if {@code depEntry} was not done.
   * Notifies the {@link GraphInconsistencyReceiver} if so. Schedules {@code depEntry} for
   * evaluation if necessary.
   *
   * <p>If {@code depEntry} was done, then this calls {@code entry.signalDep}.
   *
   * <p>If the call to {@code #signalDep} returns false, this returns {@link
   * MaybeHandleUndoneDepResult#DEP_DONE_SELF_NOT_SIGNALLED}.
   *
   * <p>If the call to {@code #signalDep} returns true, this returns {@link
   * MaybeHandleUndoneDepResult#DEP_DONE_SELF_SIGNALLED}. This will happen for the last new dep if
   * all of them were done. It can also happen if some new deps weren't done but they all signal
   * {@code entry} before {@link #maybeHandleRegisteringNewlyDiscoveredDepsForDoneEntry} finishes
   * checking deps.
   */
  private MaybeHandleUndoneDepResult maybeHandleUndoneDepForDoneEntry(
      NodeEntry entry, NodeEntry depEntry, DependencyState triState, SkyKey skyKey, SkyKey depKey) {
    if (triState == DependencyState.DONE) {
      return entry.signalDep(depEntry.getVersion(), depKey)
          ? MaybeHandleUndoneDepResult.DEP_DONE_SELF_SIGNALLED
          : MaybeHandleUndoneDepResult.DEP_DONE_SELF_NOT_SIGNALLED;
    }
    // The dep may have transitioned from done to dirty between when this node read its value and
    // now. Notify the graph inconsistency receiver so that we can crash if that's unexpected. We
    // schedule the dep if it needs scheduling, because nothing else can if we don't.
    evaluatorContext
        .getGraphInconsistencyReceiver()
        .noteInconsistencyAndMaybeThrow(
            skyKey, ImmutableList.of(depKey), Inconsistency.BUILDING_PARENT_FOUND_UNDONE_CHILD);
    if (triState == DependencyState.NEEDS_SCHEDULING) {
      // Top priority since this depKey was already evaluated before, and we want to finish it off
      // again, reducing the chance that another node may observe this dep to be undone.
      evaluatorContext.getVisitor().enqueueEvaluation(depKey, Integer.MAX_VALUE);
    }
    return MaybeHandleUndoneDepResult.DEP_NOT_DONE;
  }

  /**
   * Return true if the entry does not need to be re-evaluated this build. The entry will need to be
   * re-evaluated if it is not done, but also if it was not completely evaluated last build and this
   * build is keepGoing.
   */
  static boolean isDoneForBuild(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }
}
