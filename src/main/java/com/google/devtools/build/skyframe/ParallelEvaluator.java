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

import com.google.common.base.Function;
import com.google.common.base.Predicate;
import com.google.common.base.Suppliers;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.ProfilerTask;
import com.google.devtools.build.lib.util.BlazeClock;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.lib.util.GroupedList.GroupedListHelper;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.EvaluationProgressReceiver.EvaluationState;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.NodeEntry.DependencyState;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunctionException.ReifiedSkyFunctionException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Deque;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ForkJoinPool;
import java.util.logging.Logger;
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
 * <p>A node can be in one of three states: ready, waiting, and done. A node is ready if and only
 * if all of its dependencies have been signaled. A node is done if it has a value. It is waiting
 * if not all of its dependencies have been signaled.
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
 * the same data as before, its version (mtime) remains the same. If all of a node's children's
 * have the same version as before, its re-evaluation can be skipped.
 *
 * <p>This class is not intended for direct use, and is only exposed as public for use in
 * evaluation implementations outside of this package.
 */
public final class ParallelEvaluator implements Evaluator {

  private static final Logger LOG = Logger.getLogger(ParallelEvaluator.class.getName());

  private final ProcessableGraph graph;

  /** An general interface for {@link ParallelEvaluator} to receive objects of type {@code T}. */
  public interface Receiver<T> {
    // TODO(dmarting): should we just make it a common object for all Bazel codebase?
    /**
     * Consumes the given object.
     */
    void accept(T object);
  }

  private final DirtyKeyTracker dirtyKeyTracker;
  private final Receiver<Collection<SkyKey>> inflightKeysReceiver;

  private final ParallelEvaluatorContext evaluatorContext;

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final EventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      boolean keepGoing,
      int threadCount,
      @Nullable EvaluationProgressReceiver progressReceiver,
      DirtyKeyTracker dirtyKeyTracker,
      Receiver<Collection<SkyKey>> inflightKeysReceiver) {
    this.graph = graph;
    this.inflightKeysReceiver = inflightKeysReceiver;
    this.dirtyKeyTracker = Preconditions.checkNotNull(dirtyKeyTracker);
    evaluatorContext =
        new ParallelEvaluatorContext(
            graph,
            graphVersion,
            skyFunctions,
            reporter,
            emittedEventState,
            keepGoing,
            /*storeErrorsAlongsideValues=*/ true,
            progressReceiver,
            storedEventFilter,
            dirtyKeyTracker,
            createEvaluateRunnable(),
            threadCount);
  }

  public ParallelEvaluator(
      ProcessableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      final EventHandler reporter,
      EmittedEventState emittedEventState,
      EventFilter storedEventFilter,
      boolean keepGoing,
      boolean storeErrorsAlongsideValues,
      @Nullable EvaluationProgressReceiver progressReceiver,
      DirtyKeyTracker dirtyKeyTracker,
      Receiver<Collection<SkyKey>> inflightKeysReceiver,
      ForkJoinPool forkJoinPool) {
    this.graph = graph;
    this.inflightKeysReceiver = inflightKeysReceiver;
    Preconditions.checkState(storeErrorsAlongsideValues || keepGoing);
    this.dirtyKeyTracker = Preconditions.checkNotNull(dirtyKeyTracker);
    evaluatorContext =
        new ParallelEvaluatorContext(
            graph,
            graphVersion,
            skyFunctions,
            reporter,
            emittedEventState,
            keepGoing,
            storeErrorsAlongsideValues,
            progressReceiver,
            storedEventFilter,
            dirtyKeyTracker,
            createEvaluateRunnable(),
            Preconditions.checkNotNull(forkJoinPool));
  }

  /**
   * Creates a {@link Runnable} that is injected into the {@link NodeEntryVisitor} created on demand
   * in {@link #evaluatorContext}, so that the visitor can enqueue the appropriate {@link Runnable}
   * when it is given a {@link SkyKey} to evaluate.
   */
  private Function<SkyKey, Runnable> createEvaluateRunnable() {
    return new Function<SkyKey, Runnable>() {
      @Override
      public Runnable apply(SkyKey skyKey) {
        return new Evaluate(skyKey);
      }
    };
  }

  /**
   * If the entry is dirty and not already rebuilding, puts it in a state so that it can rebuild.
   */
  private static void maybeMarkRebuilding(NodeEntry entry) {
    if (entry.isDirty() && entry.getDirtyState() != DirtyState.REBUILDING) {
      entry.markRebuilding();
    }
  }

  private enum DirtyOutcome {
    ALREADY_PROCESSED,
    NEEDS_EVALUATION
  }

  /**
   * An action that evaluates a value.
   */
  private class Evaluate implements Runnable {
    /** The name of the value to be evaluated. */
    private final SkyKey skyKey;

    private Evaluate(SkyKey skyKey) {
      this.skyKey = skyKey;
    }

    private void enqueueChild(
        SkyKey skyKey,
        NodeEntry entry,
        SkyKey child,
        NodeEntry childEntry,
        boolean depAlreadyExists) {
      Preconditions.checkState(!entry.isDone(), "%s %s", skyKey, entry);
      DependencyState dependencyState =
          depAlreadyExists
              ? childEntry.checkIfDoneForDirtyReverseDep(skyKey)
              : childEntry.addReverseDepAndCheckIfDone(skyKey);
      switch (dependencyState) {
        case DONE:
          if (entry.signalDep(childEntry.getVersion())) {
            // This can only happen if there are no more children to be added.
            evaluatorContext.getVisitor().enqueueEvaluation(skyKey);
          }
          break;
        case ALREADY_EVALUATING:
          break;
        case NEEDS_SCHEDULING:
          evaluatorContext.getVisitor().enqueueEvaluation(child);
          break;
      }
    }

    /**
     * Returns true if this depGroup consists of the error transience value and the error transience
     * value is newer than the entry, meaning that the entry must be re-evaluated.
     */
    private boolean invalidatedByErrorTransience(Collection<SkyKey> depGroup, NodeEntry entry)
        throws InterruptedException {
      return depGroup.size() == 1
          && depGroup.contains(ErrorTransienceValue.KEY)
          && !graph.get(
              null, Reason.OTHER, ErrorTransienceValue.KEY).getVersion().atMost(entry.getVersion());
    }

    private DirtyOutcome maybeHandleDirtyNode(NodeEntry state) throws InterruptedException {
      if (!state.isDirty()) {
        return DirtyOutcome.NEEDS_EVALUATION;
      }
      switch (state.getDirtyState()) {
        case CHECK_DEPENDENCIES:
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
          Collection<SkyKey> directDepsToCheck = state.getNextDirtyDirectDeps();

          if (invalidatedByErrorTransience(directDepsToCheck, state)) {
            // If this dep is the ErrorTransienceValue and the ErrorTransienceValue has been
            // updated then we need to force a rebuild. We would like to just signal the entry as
            // usual, but we can't, because then the ErrorTransienceValue would remain as a dep,
            // which would be incorrect if, for instance, the value re-evaluated to a non-error.
            state.forceRebuild();
            graph.get(
                skyKey, Reason.RDEP_REMOVAL, ErrorTransienceValue.KEY).removeReverseDep(skyKey);
            return DirtyOutcome.NEEDS_EVALUATION;
          }
          if (!evaluatorContext.keepGoing()) {
            // This check ensures that we maintain the invariant that if a node with an error is
            // reached during a no-keep-going build, none of its currently building parents
            // finishes building. If the child isn't done building yet, it will detect on its own
            // that it has an error (see the VERIFIED_CLEAN case below). On the other hand, if it
            // is done, then it is the parent's responsibility to notice that, which we do here.
            // We check the deps for errors so that we don't continue building this node if it has
            // a child error.
            Map<SkyKey, ? extends NodeEntry> entriesToCheck =
                graph.getBatch(skyKey, Reason.OTHER, directDepsToCheck);
            for (Entry<SkyKey, ? extends NodeEntry> entry : entriesToCheck.entrySet()) {
              if (entry.getValue().isDone() && entry.getValue().getErrorInfo() != null) {
                // If any child has an error, we arbitrarily add a dep on the first one (needed
                // for error bubbling) and throw an exception coming from it.
                SkyKey errorKey = entry.getKey();
                NodeEntry errorEntry = entry.getValue();
                state.addTemporaryDirectDeps(GroupedListHelper.create(ImmutableList.of(errorKey)));
                errorEntry.checkIfDoneForDirtyReverseDep(skyKey);
                // Perform the necessary bookkeeping for any deps that are not being used.
                for (Entry<SkyKey, ? extends NodeEntry> depEntry : entriesToCheck.entrySet()) {
                  if (!depEntry.getKey().equals(errorKey)) {
                    depEntry.getValue().removeReverseDep(skyKey);
                  }
                }
                if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
                  // An error was already thrown in the evaluator. Don't do anything here.
                  return DirtyOutcome.ALREADY_PROCESSED;
                }
                throw SchedulerException.ofError(errorEntry.getErrorInfo(), entry.getKey());
              }
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

          // TODO(bazel-team): If this signals the current node, consider falling through to the
          // VERIFIED_CLEAN case below directly, without scheduling a new Evaluate().
          for (Map.Entry<SkyKey, ? extends NodeEntry> e :
              graph
                  .createIfAbsentBatch(skyKey, Reason.ENQUEUING_CHILD, directDepsToCheck)
                  .entrySet()) {
            SkyKey directDep = e.getKey();
            NodeEntry directDepEntry = e.getValue();
            enqueueChild(skyKey, state, directDep, directDepEntry, /*depAlreadyExists=*/ true);
          }
          return DirtyOutcome.ALREADY_PROCESSED;
        case VERIFIED_CLEAN:
          // No child has a changed value. This node can be marked done and its parents signaled
          // without any re-evaluation.
          evaluatorContext.getVisitor().notifyDone(skyKey);
          Set<SkyKey> reverseDeps = state.markClean();
          if (evaluatorContext.getProgressReceiver() != null) {
            // Tell the receiver that the value was not actually changed this run.
            evaluatorContext
                .getProgressReceiver()
                .evaluated(skyKey, new SkyValueSupplier(state), EvaluationState.CLEAN);
          }
          if (!evaluatorContext.keepGoing() && state.getErrorInfo() != null) {
            if (!evaluatorContext.getVisitor().preventNewEvaluations()) {
              return DirtyOutcome.ALREADY_PROCESSED;
            }
            throw SchedulerException.ofError(state.getErrorInfo(), skyKey);
          }
          evaluatorContext.signalValuesAndEnqueueIfReady(
              skyKey, reverseDeps, state.getVersion(), /*enqueueParents=*/ true);
          return DirtyOutcome.ALREADY_PROCESSED;
        case NEEDS_REBUILDING:
          maybeMarkRebuilding(state);
          // Fall through to REBUILDING case.
        case REBUILDING:
          return DirtyOutcome.NEEDS_EVALUATION;
        default:
          throw new IllegalStateException("key: " + skyKey + ", entry: " + state);
      }
    }

    @Override
    public void run() {
      try {
      NodeEntry state = Preconditions.checkNotNull(
          graph.get(null, Reason.EVALUATION, skyKey),
          skyKey);
      Preconditions.checkState(state.isReady(), "%s %s", skyKey, state);
      if (maybeHandleDirtyNode(state) == DirtyOutcome.ALREADY_PROCESSED) {
        return;
      }

      Set<SkyKey> oldDeps = state.getAllRemainingDirtyDirectDeps();
        SkyFunctionEnvironment env =
            new SkyFunctionEnvironment(
                skyKey, state.getTemporaryDirectDeps(), oldDeps, evaluatorContext);
      SkyFunctionName functionName = skyKey.functionName();
      SkyFunction factory =
          Preconditions.checkNotNull(
              evaluatorContext.getSkyFunctions().get(functionName),
              "Unable to find SkyFunction '%s' for node with key %s, %s",
              functionName,
              skyKey,
              state);

      SkyValue value = null;
      long startTime = BlazeClock.instance().nanoTime();
      try {
        value = factory.compute(skyKey, env);
      } catch (final SkyFunctionException builderException) {
        ReifiedSkyFunctionException reifiedBuilderException =
            new ReifiedSkyFunctionException(builderException, skyKey);
        // In keep-going mode, we do not let SkyFunctions throw errors with missing deps -- we will
        // restart them when their deps are done, so we can have a definitive error and definitive
        // graph structure, thus avoiding non-determinism. It's completely reasonable for
        // SkyFunctions to throw eagerly because they do not know if they are in keep-going mode.
        // Propagated transitive errors are treated the same as missing deps.
        if ((!evaluatorContext.keepGoing() || !env.valuesMissing())
            && reifiedBuilderException.getRootCauseSkyKey().equals(skyKey)) {
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
            }
          }

          Map<SkyKey, ? extends NodeEntry> newlyRequestedDeps =
              evaluatorContext.getBatchValues(
                  skyKey, Reason.RDEP_ADDITION, env.getNewlyRequestedDeps());
          boolean isTransitivelyTransient = reifiedBuilderException.isTransient();
          for (NodeEntry depEntry :
              Iterables.concat(env.getDirectDepsValues(), newlyRequestedDeps.values())) {
            if (!isDoneForBuild(depEntry)) {
              continue;
            }
            ErrorInfo depError = depEntry.getErrorInfo();
            if (depError != null) {
              isTransitivelyTransient |= depError.isTransient();
            }
          }
          ErrorInfo errorInfo = ErrorInfo.fromException(reifiedBuilderException,
              isTransitivelyTransient);
          registerNewlyDiscoveredDepsForDoneEntry(skyKey, state, newlyRequestedDeps, oldDeps, env);
          env.setError(
              state,
              errorInfo,
              /*isDirectlyTransient=*/ reifiedBuilderException.isTransient());
          env.commit(state, /*enqueueParents=*/ evaluatorContext.keepGoing());
          if (!shouldFailFast) {
            return;
          }
          throw SchedulerException.ofError(errorInfo, skyKey);
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
        long elapsedTimeNanos =  BlazeClock.instance().nanoTime() - startTime;
        if (elapsedTimeNanos > 0)  {
          if (evaluatorContext.getProgressReceiver() != null) {
            evaluatorContext.getProgressReceiver().computed(skyKey, elapsedTimeNanos);
          }
          Profiler.instance().logSimpleTaskDuration(startTime, elapsedTimeNanos,
              ProfilerTask.SKYFUNCTION, skyKey);
        }
      }

      GroupedListHelper<SkyKey> newDirectDeps = env.getNewlyRequestedDeps();

      if (value != null) {
        Preconditions.checkState(!env.valuesMissing(), "Evaluation of %s returned non-null value "
            + "but requested dependencies that weren't computed yet (one of %s), ValueEntry: %s",
            skyKey, newDirectDeps, state);
        env.setValue(value);
        registerNewlyDiscoveredDepsForDoneEntry(
            skyKey,
            state,
            graph.getBatch(skyKey, Reason.RDEP_ADDITION, env.getNewlyRequestedDeps()),
            oldDeps,
            env);
        env.commit(state, /*enqueueParents=*/true);
        return;
      }

      if (env.getDepErrorKey() != null) {
        Preconditions.checkState(
            !evaluatorContext.keepGoing(), "%s %s %s", skyKey, state, env.getDepErrorKey());
        // We encountered a child error in noKeepGoing mode, so we want to fail fast. But we first
        // need to add the edge between the current node and the child error it requested so that
        // error bubbling can occur. Note that this edge will subsequently be removed during graph
        // cleaning (since the current node will never be committed to the graph).
        SkyKey childErrorKey = env.getDepErrorKey();
        NodeEntry childErrorEntry = Preconditions.checkNotNull(
            graph.get(skyKey, Reason.OTHER, childErrorKey),
            "skyKey: %s, state: %s childErrorKey: %s",
            skyKey,
            state,
            childErrorKey);
          if (newDirectDeps.contains(childErrorKey)) {
            // Add this dep if it was just requested. In certain rare race conditions (see
            // MemoizingEvaluatorTest.cachedErrorCausesRestart) this dep may have already been
            // requested.
            state.addTemporaryDirectDeps(GroupedListHelper.create(ImmutableList.of(childErrorKey)));
            DependencyState childErrorState;
            if (oldDeps.contains(childErrorKey)) {
              childErrorState = childErrorEntry.checkIfDoneForDirtyReverseDep(skyKey);
            } else {
              childErrorState = childErrorEntry.addReverseDepAndCheckIfDone(skyKey);
            }
            Preconditions.checkState(
                childErrorState == DependencyState.DONE,
                "skyKey: %s, state: %s childErrorKey: %s",
                skyKey,
                state,
                childErrorKey,
                childErrorEntry);
        }
        ErrorInfo childErrorInfo = Preconditions.checkNotNull(childErrorEntry.getErrorInfo());
          evaluatorContext.getVisitor().preventNewEvaluations();
        throw SchedulerException.ofError(childErrorInfo, childErrorKey);
      }

      // TODO(bazel-team): This code is not safe to interrupt, because we would lose the state in
      // newDirectDeps.

      // TODO(bazel-team): An ill-behaved SkyFunction can throw us into an infinite loop where we
      // add more dependencies on every run. [skyframe-core]

      // Add all new keys to the set of known deps.
      state.addTemporaryDirectDeps(newDirectDeps);

      // If there were no newly requested dependencies, at least one of them was in error or there
      // is a bug in the SkyFunction implementation. The environment has collected its errors, so we
      // just order it to be built.
      if (newDirectDeps.isEmpty()) {
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
        env.commit(state, /*enqueueParents=*/ true);
        return;
      }

        for (Entry<SkyKey, ? extends NodeEntry> e :
            graph.createIfAbsentBatch(skyKey, Reason.ENQUEUING_CHILD, newDirectDeps).entrySet()) {
        SkyKey newDirectDep = e.getKey();
        NodeEntry newDirectDepEntry = e.getValue();
        enqueueChild(
            skyKey,
            state,
            newDirectDep,
            newDirectDepEntry,
            /*depAlreadyExists=*/ oldDeps.contains(newDirectDep));
      }
      } catch (InterruptedException ie) {
        // InterruptedException cannot be thrown by Runnable.run, so we must wrap it.
        // Interrupts can be caught by both the Evaluator and the AbstractQueueVisitor.
        // The former will unwrap the IE and propagate it as is; the latter will throw a new IE.
        throw SchedulerException.ofInterruption(ie, skyKey);
      }
      // It is critical that there is no code below this point.
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

  /**
   * If child is not done, removes {@param inProgressParent} from {@param child}'s reverse deps.
   * Returns whether child should be removed from inProgressParent's entry's direct deps.
   */
  private boolean removeIncompleteChildForCycle(SkyKey inProgressParent, SkyKey child)
      throws InterruptedException {
    NodeEntry childEntry = graph.get(inProgressParent, Reason.CYCLE_CHECKING, child);
    if (!isDoneForBuild(childEntry)) {
      childEntry.removeInProgressReverseDep(inProgressParent);
      return true;
    }
    return false;
  }

  /**
   * Add any additional deps that were registered during the run of a builder that finished by
   * creating a node or throwing an error. Builders may throw errors even if all their deps were not
   * provided -- we trust that a SkyFunction may be know it should throw an error even if not all of
   * its requested deps are done. However, that means we're assuming the SkyFunction would throw
   * that same error if all of its requested deps were done. Unfortunately, there is no way to
   * enforce that condition.
   */
  private static void registerNewlyDiscoveredDepsForDoneEntry(
      SkyKey skyKey,
      NodeEntry entry,
      Map<SkyKey, ? extends NodeEntry> newlyRequestedDepMap,
      Set<SkyKey> oldDeps,
      SkyFunctionEnvironment env) {
    Set<SkyKey> unfinishedDeps = new HashSet<>();
    for (SkyKey dep : env.getNewlyRequestedDeps()) {
      if (!isDoneForBuild(newlyRequestedDepMap.get(dep))) {
        unfinishedDeps.add(dep);
      }
    }
    env.getNewlyRequestedDeps().remove(unfinishedDeps);
    entry.addTemporaryDirectDeps(env.getNewlyRequestedDeps());
    for (SkyKey newDep : env.getNewlyRequestedDeps()) {
      // Note that this depEntry can't be null. If env.newlyRequestedDeps contained a key with a
      // null entry, then it would have been added to unfinishedDeps and then removed from
      // env.newlyRequestedDeps just above this loop.
      NodeEntry depEntry = Preconditions.checkNotNull(newlyRequestedDepMap.get(newDep), newDep);
      DependencyState triState =
          oldDeps.contains(newDep)
              ? depEntry.checkIfDoneForDirtyReverseDep(skyKey)
              : depEntry.addReverseDepAndCheckIfDone(skyKey);
      Preconditions.checkState(DependencyState.DONE == triState,
          "new dep %s was not already done for %s. ValueEntry: %s. DepValueEntry: %s",
          newDep, skyKey, entry, depEntry);
      entry.signalDep();
    }
    Preconditions.checkState(
        entry.isReady(), "%s %s %s", skyKey, entry, env.getNewlyRequestedDeps());
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
  public <T extends SkyValue> EvaluationResult<T> eval(Iterable<SkyKey> skyKeys)
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
    NodeEntry errorTransienceEntry = Iterables.getOnlyElement(graph.createIfAbsentBatch(
            null,
            Reason.PRE_OR_POST_EVALUATION,
            ImmutableList.of(ErrorTransienceValue.KEY)).values());
    if (!errorTransienceEntry.isDone()) {
      injectValues(
          ImmutableMap.of(ErrorTransienceValue.KEY, (SkyValue) ErrorTransienceValue.INSTANCE),
          evaluatorContext.getGraphVersion(),
          graph,
          dirtyKeyTracker);
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
    try {
      return waitForCompletionAndConstructResult(skyKeys);
    } finally {
      inflightKeysReceiver.accept(evaluatorContext.getVisitor().getInflightNodes());
    }
  }

  private <T extends SkyValue> EvaluationResult<T> waitForCompletionAndConstructResult(
      Iterable<SkyKey> skyKeys) throws InterruptedException {
    Map<SkyKey, ValueWithMetadata> bubbleErrorInfo = null;
    boolean catastrophe = false;
    try {
      evaluatorContext.getVisitor().waitForCompletion();
    } catch (final SchedulerException e) {
      if (!evaluatorContext.getVisitor().getCrashes().isEmpty()) {
        evaluatorContext
            .getReporter()
            .handle(Event.error("Crashes detected: " + evaluatorContext.getVisitor().getCrashes()));
        throw Iterables.getFirst(evaluatorContext.getVisitor().getCrashes(), null);
      }
      Throwables.propagateIfPossible(e.getCause(), InterruptedException.class);
      if (Thread.interrupted()) {
        // As per the contract of AbstractQueueVisitor#work, if an unchecked exception is thrown and
        // the build is interrupted, the thrown exception is what will be rethrown. Since the user
        // presumably wanted to interrupt the build, we ignore the thrown SchedulerException (which
        // doesn't indicate a programming bug) and throw an InterruptedException.
        throw new InterruptedException();
      }

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
      Iterable<SkyKey> reverseDeps = errorEntry.isDone()
          ? errorEntry.getReverseDeps()
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
        if (evaluatorContext.getVisitor().isInflight(bubbleParent)
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
          bubbleErrorInfo.put(errorKey,
              ValueWithMetadata.error(ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
                  env.buildEvents(parentEntry, /*missingChildren=*/true)));
          continue;
        }
      } finally {
        // Clear interrupted status. We're not listening to interrupts here.
        Thread.interrupted();
      }
      // Builder didn't throw an exception, so just propagate this one up.
      bubbleErrorInfo.put(errorKey,
          ValueWithMetadata.error(ErrorInfo.fromChildErrors(errorKey, ImmutableSet.of(error)),
              env.buildEvents(parentEntry, /*missingChildren=*/true)));
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
      checkForCycles(cycleRoots, result);
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

  private <T extends SkyValue> void checkForCycles(
      Iterable<SkyKey> badRoots, EvaluationResult.Builder<T> result) throws InterruptedException {
    try (AutoProfiler p = AutoProfiler.logged("Checking for Skyframe cycles", LOG, 10)) {
      for (SkyKey root : badRoots) {
        ErrorInfo errorInfo = checkForCycles(root);
        if (errorInfo == null) {
          // This node just wasn't finished when evaluation aborted -- there were no cycles below
          // it.
          Preconditions.checkState(!evaluatorContext.keepGoing(), "", root, badRoots);
          continue;
        }
        Preconditions.checkState(!Iterables.isEmpty(errorInfo.getCycleInfo()),
            "%s was not evaluated, but was not part of a cycle", root);
        result.addError(root, errorInfo);
        if (!evaluatorContext.keepGoing()) {
          return;
        }
      }
    }
  }

  /**
   * Marker value that we push onto a stack before we push a node's children on. When the marker
   * value is popped, we know that all the children are finished. We would use null instead, but
   * ArrayDeque does not permit null elements.
   */
  private static final SkyKey CHILDREN_FINISHED =
      SkyKey.create(SkyFunctionName.create("MARKER"), "MARKER");

  /** The max number of cycles we will report to the user for a given root, to avoid OOMing. */
  private static final int MAX_CYCLES = 20;

  /**
   * The algorithm for this cycle detector is as follows. We visit the graph depth-first, keeping
   * track of the path we are currently on. We skip any DONE nodes (they are transitively
   * error-free). If we come to a node already on the path, we immediately construct a cycle. If we
   * are in the noKeepGoing case, we return ErrorInfo with that cycle to the caller. Otherwise, we
   * continue. Once all of a node's children are done, we construct an error value for it, based on
   * those children. Finally, when the original root's node is constructed, we return its ErrorInfo.
   */
  private ErrorInfo checkForCycles(SkyKey root) throws InterruptedException {
    // The number of cycles found. Do not keep on searching for more cycles after this many were
    // found.
    int cyclesFound = 0;
    // The path through the graph currently being visited.
    List<SkyKey> graphPath = new ArrayList<>();
    // Set of nodes on the path, to avoid expensive searches through the path for cycles.
    Set<SkyKey> pathSet = new HashSet<>();

    // Maintain a stack explicitly instead of recursion to avoid stack overflows
    // on extreme graphs (with long dependency chains).
    Deque<SkyKey> toVisit = new ArrayDeque<>();

    toVisit.push(root);

    // The procedure for this check is as follows: we visit a node, push it onto the graph stack,
    // push a marker value onto the toVisit stack, and then push all of its children onto the
    // toVisit stack. Thus, when the marker node comes to the top of the toVisit stack, we have
    // visited the downward transitive closure of the value. At that point, all of its children must
    // be finished, and so we can build the definitive error info for the node, popping it off the
    // graph stack.
    while (!toVisit.isEmpty()) {
      SkyKey key = toVisit.pop();

      NodeEntry entry;
      if (key == CHILDREN_FINISHED) {
        // A marker node means we are done with all children of a node. Since all nodes have
        // errors, we must have found errors in the children when that happens.
        key = graphPath.remove(graphPath.size() - 1);
        entry = Preconditions.checkNotNull(graph.get(null, Reason.CYCLE_CHECKING, key), key);
        pathSet.remove(key);
        // Skip this node if it was first/last node of a cycle, and so has already been processed.
        if (entry.isDone()) {
          continue;
        }
        if (!evaluatorContext.keepGoing()) {
          // in the --nokeep_going mode, we would have already returned if we'd found a cycle below
          // this node. The fact that we haven't means that there were no cycles below this node
          // -- it just hadn't finished evaluating. So skip it.
          continue;
        }
        Set<SkyKey> removedDeps = ImmutableSet.of();
        if (cyclesFound < MAX_CYCLES) {
          // Value must be ready, because all of its children have finished, so we can build its
          // error.
          Preconditions.checkState(entry.isReady(), "%s not ready. ValueEntry: %s", key, entry);
        } else if (!entry.isReady()) {
          removedDeps =
              removeIncompleteChildrenForCycle(
                  key, entry, Iterables.concat(entry.getTemporaryDirectDeps()));
        }
        maybeMarkRebuilding(entry);
        GroupedList<SkyKey> directDeps = entry.getTemporaryDirectDeps();
        // Find out which children have errors. Similar logic to that in Evaluate#run().
        List<ErrorInfo> errorDeps = getChildrenErrorsForCycle(key, Iterables.concat(directDeps));
        Preconditions.checkState(!errorDeps.isEmpty(),
            "Value %s was not successfully evaluated, but had no child errors. ValueEntry: %s", key,
            entry);
        SkyFunctionEnvironment env =
            new SkyFunctionEnvironment(
                key,
                directDeps,
                Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                evaluatorContext);
        env.setError(
            entry, ErrorInfo.fromChildErrors(key, errorDeps), /*isDirectlyTransient=*/false);
        env.commit(entry, /*enqueueParents=*/false);
      } else {
        entry = graph.get(null, Reason.CYCLE_CHECKING, key);
      }

      Preconditions.checkNotNull(entry, key);
      // Nothing to be done for this node if it already has an entry.
      if (entry.isDone()) {
        continue;
      }
      if (cyclesFound == MAX_CYCLES) {
        // Do not keep on searching for cycles indefinitely, to avoid excessive runtime/OOMs.
        continue;
      }

      if (pathSet.contains(key)) {
        int cycleStart = graphPath.indexOf(key);
        // Found a cycle!
        cyclesFound++;
        Iterable<SkyKey> cycle = graphPath.subList(cycleStart, graphPath.size());
        LOG.info("Found cycle : " + cycle + " from " + graphPath);
        // Put this node into a consistent state for building if it is dirty.
        if (entry.isDirty() && entry.getDirtyState() == NodeEntry.DirtyState.CHECK_DEPENDENCIES) {
          // In the check deps state, entry has exactly one child not done yet. Note that this node
          // must be part of the path to the cycle we have found (since done nodes cannot be in
          // cycles, and this is the only missing one). Thus, it will not be removed below in
          // removeDescendantsOfCycleValue, so it is safe here to signal that it is done.
          entry.signalDep();
          maybeMarkRebuilding(entry);
        }
        if (evaluatorContext.keepGoing()) {
          // Any children of this node that we haven't already visited are not worth visiting,
          // since this node is about to be done. Thus, the only child worth visiting is the one in
          // this cycle, the cycleChild (which may == key if this cycle is a self-edge).
          SkyKey cycleChild = selectCycleChild(key, graphPath, cycleStart);
          Set<SkyKey> removedDeps =
              removeDescendantsOfCycleValue(
                  key, entry, cycleChild, toVisit, graphPath.size() - cycleStart);
          ValueWithMetadata dummyValue = ValueWithMetadata.wrapWithMetadata(new SkyValue() {});


          SkyFunctionEnvironment env =
              new SkyFunctionEnvironment(
                  key,
                  entry.getTemporaryDirectDeps(),
                  ImmutableMap.of(cycleChild, dummyValue),
                  Sets.difference(entry.getAllRemainingDirtyDirectDeps(), removedDeps),
                  evaluatorContext);

          // Construct error info for this node. Get errors from children, which are all done
          // except possibly for the cycleChild.
          List<ErrorInfo> allErrors =
              getChildrenErrorsForCycleChecking(
                  Iterables.concat(entry.getTemporaryDirectDeps()),
                  /*unfinishedChild=*/ cycleChild);
          CycleInfo cycleInfo = new CycleInfo(cycle);
          // Add in this cycle.
          allErrors.add(ErrorInfo.fromCycle(cycleInfo));
          env.setError(entry, ErrorInfo.fromChildErrors(key, allErrors), /*isTransient=*/false);
          env.commit(entry, /*enqueueParents=*/false);
          continue;
        } else {
          // We need to return right away in the noKeepGoing case, so construct the cycle (with the
          // path) and return.
          Preconditions.checkState(graphPath.get(0).equals(root),
              "%s not reached from %s. ValueEntry: %s", key, root, entry);
          return ErrorInfo.fromCycle(new CycleInfo(graphPath.subList(0, cycleStart), cycle));
        }
      }

      // This node is not yet known to be in a cycle. So process its children.
      Iterable<SkyKey> children = Iterables.concat(entry.getTemporaryDirectDeps());
      if (Iterables.isEmpty(children)) {
        continue;
      }
      // Prefetch all children, in case our graph performs better with a primed cache. No need to
      // recurse into done nodes. The fields of done nodes aren't necessary, since we'll filter them
      // out.
      // TODO(janakr): If graph implementations start using these hints for not-done nodes, we may
      // have to change this.
      Map<SkyKey, ? extends NodeEntry> childrenNodes =
          graph.getBatch(key, Reason.EXISTENCE_CHECKING, children);
      Preconditions.checkState(childrenNodes.size() == Iterables.size(children), childrenNodes);
      children = Maps.filterValues(childrenNodes, new Predicate<NodeEntry>() {
        @Override
        public boolean apply(NodeEntry nodeEntry) {
          return !nodeEntry.isDone();
        }
      }).keySet();

      // This marker flag will tell us when all this node's children have been processed.
      toVisit.push(CHILDREN_FINISHED);
      // This node is now part of the path through the graph.
      graphPath.add(key);
      pathSet.add(key);
      for (SkyKey nextValue : children) {
        toVisit.push(nextValue);
      }
    }
    return evaluatorContext.keepGoing() ? getAndCheckDoneForCycle(root).getErrorInfo() : null;
  }

  /**
   * Returns the child of this node that is in the cycle that was just found. If the cycle is a
   * self-edge, returns the node itself.
   */
  private static SkyKey selectCycleChild(SkyKey key, List<SkyKey> graphPath, int cycleStart) {
    return cycleStart + 1 == graphPath.size() ? key : graphPath.get(cycleStart + 1);
  }

  /**
   * Get all the errors of child nodes. There must be at least one cycle amongst them.
   *
   * @param children child nodes to query for errors.
   * @return List of ErrorInfos from all children that had errors.
   */
  private List<ErrorInfo> getChildrenErrorsForCycle(SkyKey parent, Iterable<SkyKey> children)
      throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    boolean foundCycle = false;
    for (NodeEntry childNode : getAndCheckDoneBatchForCycle(parent, children).values()) {
      ErrorInfo errorInfo = childNode.getErrorInfo();
      if (errorInfo != null) {
        foundCycle |= !Iterables.isEmpty(errorInfo.getCycleInfo());
        allErrors.add(errorInfo);
      }
    }
    Preconditions.checkState(foundCycle, "", children, allErrors);
    return allErrors;
  }

  /**
   * Get all the errors of child nodes.
   *
   * @param children child nodes to query for errors.
   * @param unfinishedChild child which is allowed to not be done.
   * @return List of ErrorInfos from all children that had errors.
   */
  private List<ErrorInfo> getChildrenErrorsForCycleChecking(
      Iterable<SkyKey> children, SkyKey unfinishedChild) throws InterruptedException {
    List<ErrorInfo> allErrors = new ArrayList<>();
    Set<? extends Entry<SkyKey, ? extends NodeEntry>> childEntries =
        evaluatorContext.getBatchValues(null, Reason.CYCLE_CHECKING, children).entrySet();
    for (Entry<SkyKey, ? extends NodeEntry> childMapEntry : childEntries) {
      SkyKey childKey = childMapEntry.getKey();
      NodeEntry childNodeEntry = childMapEntry.getValue();
      ErrorInfo errorInfo = getErrorMaybe(childKey, childNodeEntry,
          /*allowUnfinished=*/childKey.equals(unfinishedChild));
      if (errorInfo != null) {
        allErrors.add(errorInfo);
      }
    }
    return allErrors;
  }

  @Nullable
  private static ErrorInfo getErrorMaybe(
      SkyKey key, NodeEntry childNodeEntry, boolean allowUnfinished) throws InterruptedException {
    Preconditions.checkNotNull(childNodeEntry, key);
    if (!allowUnfinished) {
      return checkDone(key, childNodeEntry).getErrorInfo();
    }
    return childNodeEntry.isDone() ? childNodeEntry.getErrorInfo() : null;
  }

  /**
   * Removes direct children of key from toVisit and from the entry itself, and makes the entry
   * ready if necessary. We must do this because it would not make sense to try to build the
   * children after building the entry. It would violate the invariant that a parent can only be
   * built after its children are built; See bug "Precondition error while evaluating a Skyframe
   * graph with a cycle".
   *
   * @param key SkyKey of node in a cycle.
   * @param entry NodeEntry of node in a cycle.
   * @param cycleChild direct child of key in the cycle, or key itself if the cycle is a self-edge.
   * @param toVisit list of remaining nodes to visit by the cycle-checker.
   * @param cycleLength the length of the cycle found.
   */
  private Set<SkyKey> removeDescendantsOfCycleValue(
      SkyKey key,
      NodeEntry entry,
      @Nullable SkyKey cycleChild,
      Iterable<SkyKey> toVisit,
      int cycleLength)
      throws InterruptedException {
    GroupedList<SkyKey> directDeps = entry.getTemporaryDirectDeps();
    Set<SkyKey> unvisitedDeps = Sets.newHashSetWithExpectedSize(directDeps.numElements());
    Iterables.addAll(unvisitedDeps, Iterables.concat(directDeps));
    unvisitedDeps.remove(cycleChild);
    // Remove any children from this node that are not part of the cycle we just found. They are
    // irrelevant to the node as it stands, and if they are deleted from the graph because they are
    // not built by the end of cycle-checking, we would have dangling references.
    Set<SkyKey> removedDeps = removeIncompleteChildrenForCycle(key, entry, unvisitedDeps);
    if (!entry.isReady()) {
      // The entry has at most one undone dep now, its cycleChild. Signal to make entry ready. Note
      // that the entry can conceivably be ready if its cycleChild already found a different cycle
      // and was built.
      entry.signalDep();
    }
    maybeMarkRebuilding(entry);
    Preconditions.checkState(entry.isReady(), "%s %s %s", key, cycleChild, entry);
    Iterator<SkyKey> it = toVisit.iterator();
    while (it.hasNext()) {
      SkyKey descendant = it.next();
      if (descendant == CHILDREN_FINISHED) {
        // Marker value, delineating the end of a group of children that were enqueued.
        cycleLength--;
        if (cycleLength == 0) {
          // We have seen #cycleLength-1 marker values, and have arrived at the one for this value,
          // so we are done.
          return removedDeps;
        }
        continue; // Don't remove marker values.
      }
      if (cycleLength == 1) {
        // Remove the direct children remaining to visit of the cycle node.
        Preconditions.checkState(unvisitedDeps.contains(descendant),
            "%s %s %s %s %s", key, descendant, cycleChild, unvisitedDeps, entry);
        it.remove();
      }
    }
    throw new IllegalStateException("There were not " + cycleLength + " groups of children in "
        + toVisit + " when trying to remove children of " + key + " other than " + cycleChild);
  }

  private Set<SkyKey> removeIncompleteChildrenForCycle(
      SkyKey key, NodeEntry entry, Iterable<SkyKey> children) throws InterruptedException {
    Set<SkyKey> unfinishedDeps = new HashSet<>();
    for (SkyKey child : children) {
      if (removeIncompleteChildForCycle(key, child)) {
        unfinishedDeps.add(child);
      }
    }
    entry.removeUnfinishedDeps(unfinishedDeps);
    return unfinishedDeps;
  }

  private static NodeEntry checkDone(SkyKey key, NodeEntry entry) {
    Preconditions.checkNotNull(entry, key);
    Preconditions.checkState(entry.isDone(), "%s %s", key, entry);
    return entry;
  }

  private NodeEntry getAndCheckDoneForCycle(SkyKey key) throws InterruptedException {
    return checkDone(key, graph.get(null, Reason.CYCLE_CHECKING, key));
  }

  private Map<SkyKey, ? extends NodeEntry> getAndCheckDoneBatchForCycle(
      SkyKey parent, Iterable<SkyKey> keys) throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> nodes =
        evaluatorContext.getBatchValues(parent, Reason.CYCLE_CHECKING, keys);
    for (Entry<SkyKey, ? extends NodeEntry> nodeEntryMapEntry : nodes.entrySet()) {
      checkDone(nodeEntryMapEntry.getKey(), nodeEntryMapEntry.getValue());
    }
    return nodes;
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

  /**
   * Return true if the entry does not need to be re-evaluated this build. The entry will need to be
   * re-evaluated if it is not done, but also if it was not completely evaluated last build and this
   * build is keepGoing.
   */
  static boolean isDoneForBuild(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }

  static void injectValues(
      Map<SkyKey, SkyValue> injectionMap,
      Version version,
      EvaluableGraph graph,
      DirtyKeyTracker dirtyKeyTracker)
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
      dirtyKeyTracker.notDirty(key);
    }
  }
}
