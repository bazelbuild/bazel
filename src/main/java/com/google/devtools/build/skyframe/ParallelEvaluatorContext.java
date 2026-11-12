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

import static com.google.common.base.Preconditions.checkNotNull;

import com.github.benmanes.caffeine.cache.Cache;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.MultiThreadPoolsQuiescingExecutor.ThreadPoolType;
import com.google.devtools.build.lib.concurrent.QuiescingExecutor;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.Reportable;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyFunction.Environment.SkyKeyComputeState;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.function.Predicate;
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
  private final Version minimalVersion;
  private final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private final ExtendedEventHandler reporter;
  private final EmittedEventState emittedEventState;
  private final NestedSetVisitor<Reportable> replayingNestedSetEventVisitor;
  private final Predicate<SkyKey> keepGoing;

  private final InflightTrackingProgressReceiver progressReceiver;
  private final EventFilter storedEventFilter;
  private final ErrorInfoManager errorInfoManager;
  private final GraphInconsistencyReceiver graphInconsistencyReceiver;
  private final QuiescingExecutor executor;
  private final Cache<SkyKey, SkyKeyComputeState> stateCache;
  private final boolean dfsScheduling;
  @Nullable private final ConcurrentHashMap<SkyKey, Integer> nodeDepths;

  /**
   * The visitor managing the thread pool. Used to enqueue parents when an entry is finished, and,
   * during testing, to block until an exception is thrown if a node builder requests that.
   * Initialized after construction to avoid the overhead of the caller's creating a threadpool in
   * cases where it is not needed.
   */
  private final Supplier<NodeEntryVisitor> visitorSupplier;

  /** * Returns a {@link Runnable} given a {@code key} to evaluate. */
  interface RunnableMaker {
    Runnable make(SkyKey key, boolean alreadyStarted);
  }

  public ParallelEvaluatorContext(
      QueryableGraph graph,
      Version graphVersion,
      Version minimalVersion,
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      InflightTrackingProgressReceiver progressReceiver,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      QuiescingExecutor executor,
      Supplier<NodeEntryVisitor> visitorSupplier,
      Cache<SkyKey, SkyKeyComputeState> stateCache,
      Predicate<SkyKey> keepGoing) {
    this.graph = graph;
    this.graphVersion = graphVersion;
    this.minimalVersion = minimalVersion;
    this.skyFunctions = skyFunctions;
    this.reporter = reporter;
    this.graphInconsistencyReceiver = graphInconsistencyReceiver;
    this.emittedEventState = emittedEventState;
    this.replayingNestedSetEventVisitor =
        new NestedSetVisitor<>(
            new NestedSetEventReceiver(reporter), emittedEventState.asVisitedState());
    this.progressReceiver = checkNotNull(progressReceiver);
    this.storedEventFilter = storedEventFilter;
    this.errorInfoManager = errorInfoManager;
    this.executor = executor;
    this.visitorSupplier = Suppliers.memoize(visitorSupplier);
    this.stateCache = stateCache;
    this.keepGoing = keepGoing;
    this.dfsScheduling =
        executor instanceof AbstractQueueVisitor queueVisitor
            && queueVisitor.ordersTasksByPriority();
    this.nodeDepths = this.dfsScheduling ? new ConcurrentHashMap<>() : null;
  }

  /**
   * Returns whether {@code key} is scheduled on a priority-ordered thread pool.
   *
   * <p>{@link #dfsScheduling} indicates whether the analysis pool orders tasks by priority, but a
   * {@link MultiThreadPoolsQuiescingExecutor} also routes execution-phase keys to a separate
   * non-priority pool ({@link ThreadPoolType#EXECUTION_PHASE}). Checking the target pool for {@code
   * key} avoids tracking graph depths and computing priority tiers for keys whose pool ignores task
   * priority.
   */
  boolean isDfsScheduled(SkyKey key) {
    if (!dfsScheduling) {
      return false;
    }
    if (executor instanceof MultiThreadPoolsQuiescingExecutor multiPoolExecutor) {
      return NodeEntryVisitor.selectThreadPoolType(multiPoolExecutor, key)
          != ThreadPoolType.EXECUTION_PHASE;
    }
    return true;
  }

  int getNodeDepth(SkyKey key) {
    return nodeDepths != null ? nodeDepths.getOrDefault(key, /* defaultValue= */ 0) : 0;
  }

  void recordChildDepth(SkyKey child, int candidateDepth) {
    if (nodeDepths != null
        && isDfsScheduled(child)
        && nodeDepths.getOrDefault(child, /* defaultValue= */ -1) < candidateDepth) {
      nodeDepths.merge(child, candidateDepth, Math::max);
    }
  }

  void clearNodeDepth(SkyKey key) {
    if (nodeDepths != null && isDfsScheduled(key)) {
      nodeDepths.remove(key);
    }
  }

  /**
   * Signals all parents that this node is finished.
   *
   * <p>Calling this method indicates that we are building this node after the main build aborted,
   * so skips signalling any parents that are already done (that can happen with cycles).
   */
  void signalParentsOnAbort(SkyKey skyKey, Set<SkyKey> parents, Version version)
      throws InterruptedException {
    NodeBatch batch = graph.getBatch(skyKey, Reason.SIGNAL_DEP, parents);
    for (SkyKey parent : parents) {
      NodeEntry entry = checkNotNull(batch.get(parent), parent);
      if (!entry.isDone()) { // In cycles, we can have parents that are already done.
        entry.signalDep(version, skyKey);
      }
    }
  }

  /** Signals all parents that this node is finished and enqueues any parents that are ready. */
  void signalParentsAndEnqueueIfReady(SkyKey skyKey, Set<SkyKey> parents, Version version)
      throws InterruptedException {
    NodeBatch batch = graph.getBatch(skyKey, Reason.SIGNAL_DEP, parents);
    for (SkyKey parent : parents) {
      NodeEntry entry = checkNotNull(batch.get(parent), parent);
      boolean evaluationRequired = entry.signalDep(version, skyKey);
      if (evaluationRequired || parent.supportsPartialReevaluation()) {
        getVisitor().enqueueEvaluation(parent, skyKey, /* alreadyStarted= */ true);
      }
    }
  }

  QueryableGraph getGraph() {
    return graph;
  }

  Version getGraphVersion() {
    return graphVersion;
  }

  Version getMinimalVersion() {
    return minimalVersion;
  }

  boolean keepGoing(SkyKey key) {
    return keepGoing.test(key);
  }

  NodeEntryVisitor getVisitor() {
    return visitorSupplier.get();
  }

  InflightTrackingProgressReceiver getProgressReceiver() {
    return progressReceiver;
  }

  GraphInconsistencyReceiver getGraphInconsistencyReceiver() {
    return graphInconsistencyReceiver;
  }

  EmittedEventState getEmittedEventState() {
    return emittedEventState;
  }

  NestedSetVisitor<Reportable> getReplayingNestedSetEventVisitor() {
    return replayingNestedSetEventVisitor;
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

  QuiescingExecutor getExecutor() {
    return executor;
  }

  Cache<SkyKey, SkyKeyComputeState> stateCache() {
    return stateCache;
  }

  /** Receives the events from the NestedSet and delegates to the reporter. */
  private static final class NestedSetEventReceiver
      implements NestedSetVisitor.Receiver<Reportable> {
    private final ExtendedEventHandler reporter;

    NestedSetEventReceiver(ExtendedEventHandler reporter) {
      this.reporter = reporter;
    }

    @Override
    public void accept(Reportable event) {
      event.reportTo(reporter);
    }
  }
}
