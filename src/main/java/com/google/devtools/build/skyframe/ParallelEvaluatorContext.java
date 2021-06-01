// Copyright 2016 The Bazel Authors. All rights reserved.
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
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Context object holding sufficient information for {@link SkyFunctionEnvironment} to perform its
 * duties. Shared among all {@link SkyFunctionEnvironment} instances, which should regard this
 * object as a read-only collection of data.
 *
 * <p>Also used during cycle detection.
 */
class ParallelEvaluatorContext {

  private final QueryableGraph graph;
  private final Version graphVersion;
  private final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private final ExtendedEventHandler reporter;
  private final NestedSetVisitor<TaggedEvents> replayingNestedSetEventVisitor;
  private final NestedSetVisitor<Postable> replayingNestedSetPostableVisitor;
  private final boolean keepGoing;
  private final DirtyTrackingProgressReceiver progressReceiver;
  private final EventFilter storedEventFilter;
  private final ErrorInfoManager errorInfoManager;
  private final GraphInconsistencyReceiver graphInconsistencyReceiver;
  private final EvaluationVersionBehavior evaluationVersionBehavior;

  /**
   * The visitor managing the thread pool. Used to enqueue parents when an entry is finished, and,
   * during testing, to block until an exception is thrown if a node builder requests that.
   * Initialized after construction to avoid the overhead of the caller's creating a threadpool in
   * cases where it is not needed.
   */
  private final Supplier<NodeEntryVisitor> visitorSupplier;

  /**
   * Returns a {@link Runnable} given a {@code key} to evaluate and an {@code evaluationPriority}
   * indicating whether it should be scheduled for evaluation soon (higher is better). The returned
   * {@link Runnable} is a {@link ComparableRunnable} so that it can be ordered by {@code
   * evaluationPriority} in a priority queue if needed.
   */
  interface RunnableMaker {
    ComparableRunnable make(SkyKey key, int evaluationPriority);
  }

  interface ComparableRunnable extends Runnable, Comparable<ComparableRunnable> {}

  public ParallelEvaluatorContext(
      QueryableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      boolean keepGoing,
      DirtyTrackingProgressReceiver progressReceiver,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      Supplier<NodeEntryVisitor> visitorSupplier,
      EvaluationVersionBehavior evaluationVersionBehavior) {
    this.graph = graph;
    this.graphVersion = graphVersion;
    this.skyFunctions = skyFunctions;
    this.reporter = reporter;
    this.graphInconsistencyReceiver = graphInconsistencyReceiver;
    this.evaluationVersionBehavior = evaluationVersionBehavior;
    this.replayingNestedSetEventVisitor =
        new NestedSetVisitor<>(new NestedSetEventReceiver(reporter), emittedEventState.eventState);
    this.replayingNestedSetPostableVisitor =
        new NestedSetVisitor<>(
            new NestedSetPostableReceiver(reporter), emittedEventState.postableState);
    this.keepGoing = keepGoing;
    this.progressReceiver = Preconditions.checkNotNull(progressReceiver);
    this.storedEventFilter = storedEventFilter;
    this.errorInfoManager = errorInfoManager;
    this.visitorSupplier = Suppliers.memoize(visitorSupplier);
  }

  Map<SkyKey, ? extends NodeEntry> getBatchValues(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
      throws InterruptedException {
    return graph.getBatch(requestor, reason, keys);
  }

  /**
   * Signals all parents that this node is finished.
   *
   * <p>Calling this method indicates that we are building this node after the main build aborted,
   * so skips signalling any parents that are already done (that can happen with cycles).
   */
  void signalParentsOnAbort(SkyKey skyKey, Iterable<SkyKey> parents, Version version)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> batch = getBatchValues(skyKey, Reason.SIGNAL_DEP, parents);
    for (SkyKey parent : parents) {
      NodeEntry entry = Preconditions.checkNotNull(batch.get(parent), parent);
      if (!entry.isDone()) { // In cycles, we can have parents that are already done.
        entry.signalDep(version, skyKey);
      }
    }
  }

  /**
   * Signals all parents that this node is finished and enqueues any parents that are ready at the
   * given evaluation priority.
   */
  void signalParentsAndEnqueueIfReady(
      SkyKey skyKey, Iterable<SkyKey> parents, Version version, int evaluationPriority)
      throws InterruptedException {
    Map<SkyKey, ? extends NodeEntry> batch = getBatchValues(skyKey, Reason.SIGNAL_DEP, parents);
    for (SkyKey parent : parents) {
      NodeEntry entry = Preconditions.checkNotNull(batch.get(parent), parent);
      if (entry.signalDep(version, skyKey)) {
        getVisitor().enqueueEvaluation(parent, evaluationPriority);
      }
    }
  }

  QueryableGraph getGraph() {
    return graph;
  }

  Version getGraphVersion() {
    return graphVersion;
  }

  boolean keepGoing() {
    return keepGoing;
  }

  NodeEntryVisitor getVisitor() {
    return visitorSupplier.get();
  }

  DirtyTrackingProgressReceiver getProgressReceiver() {
    return progressReceiver;
  }

  GraphInconsistencyReceiver getGraphInconsistencyReceiver() {
    return graphInconsistencyReceiver;
  }

  NestedSetVisitor<TaggedEvents> getReplayingNestedSetEventVisitor() {
    return replayingNestedSetEventVisitor;
  }

  NestedSetVisitor<Postable> getReplayingNestedSetPostableVisitor() {
    return replayingNestedSetPostableVisitor;
  }

  ExtendedEventHandler getReporter() {
    return reporter;
  }

  ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctions() {
    return skyFunctions;
  }

  EventFilter getStoredEventFilter() {
    return storedEventFilter;
  }

  ErrorInfoManager getErrorInfoManager() {
    return errorInfoManager;
  }

  EvaluationVersionBehavior getEvaluationVersionBehavior() {
    return evaluationVersionBehavior;
  }

  boolean restartPermitted() {
    return graphInconsistencyReceiver.restartPermitted();
  }

  /** Receives the events from the NestedSet and delegates to the reporter. */
  private static final class NestedSetEventReceiver
      implements NestedSetVisitor.Receiver<TaggedEvents> {
    private final ExtendedEventHandler reporter;

    NestedSetEventReceiver(ExtendedEventHandler reporter) {
      this.reporter = reporter;
    }

    @Override
    public void accept(TaggedEvents events) {
      for (Event e : events.getEvents()) {
        reporter.handle(e);
      }
    }
  }

  /** Receives the postables from the NestedSet and delegates to the reporter. */
  private static final class NestedSetPostableReceiver
      implements NestedSetVisitor.Receiver<Postable> {
    private final ExtendedEventHandler reporter;

    NestedSetPostableReceiver(ExtendedEventHandler reporter) {
      this.reporter = reporter;
    }

    @Override
    public void accept(Postable post) {
      reporter.post(post);
    }
  }
}
