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

import com.google.common.base.Function;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.collect.nestedset.NestedSetVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.ExtendedEventHandler;
import com.google.devtools.build.lib.events.ExtendedEventHandler.Postable;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.MemoizingEvaluator.EmittedEventState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.Map;
import java.util.concurrent.ForkJoinPool;
import javax.annotation.Nullable;

/**
 * Context object holding sufficient information for {@link SkyFunctionEnvironment} to perform its
 * duties. Shared among all {@link SkyFunctionEnvironment} instances, which should regard this
 * object as a read-only collection of data.
 *
 * <p>Also used during cycle detection.
 */
class ParallelEvaluatorContext {
  enum EnqueueParentBehavior {
    ENQUEUE,
    SIGNAL,
    NO_ACTION
  }

  private final QueryableGraph graph;
  private final Version graphVersion;
  private final ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions;
  private final ExtendedEventHandler reporter;
  private final NestedSetVisitor<TaggedEvents> replayingNestedSetEventVisitor;
  private final NestedSetVisitor<Postable> replayingNestedSetPostableVisitor;
  private final boolean keepGoing;
  private final DirtyTrackingProgressReceiver progressReceiver;
  private final EventFilter storedEventFilter;
  private final ErrorInfoManager errorInfoManager;

  /**
   * The visitor managing the thread pool. Used to enqueue parents when an entry is finished, and,
   * during testing, to block until an exception is thrown if a node builder requests that.
   * Initialized after construction to avoid the overhead of the caller's creating a threadpool in
   * cases where it is not needed.
   */
  private final Supplier<NodeEntryVisitor> visitorSupplier;

  ParallelEvaluatorContext(
      QueryableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      boolean keepGoing,
      final DirtyTrackingProgressReceiver progressReceiver,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      final Function<SkyKey, Runnable> runnableMaker,
      final int threadCount) {
    this.graph = graph;
    this.graphVersion = graphVersion;
    this.skyFunctions = skyFunctions;
    this.reporter = reporter;
    this.replayingNestedSetEventVisitor =
        new NestedSetVisitor<>(new NestedSetEventReceiver(reporter), emittedEventState.eventState);
    this.replayingNestedSetPostableVisitor =
        new NestedSetVisitor<>(
            new NestedSetPostableReceiver(reporter), emittedEventState.postableState);
    this.keepGoing = keepGoing;
    this.progressReceiver = Preconditions.checkNotNull(progressReceiver);
    this.storedEventFilter = storedEventFilter;
    this.errorInfoManager = errorInfoManager;
    visitorSupplier =
        Suppliers.memoize(
            new Supplier<NodeEntryVisitor>() {
              @Override
              public NodeEntryVisitor get() {
                return new NodeEntryVisitor(
                    threadCount, progressReceiver, runnableMaker);
              }
            });
  }

  ParallelEvaluatorContext(
      QueryableGraph graph,
      Version graphVersion,
      ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      ExtendedEventHandler reporter,
      EmittedEventState emittedEventState,
      boolean keepGoing,
      final DirtyTrackingProgressReceiver progressReceiver,
      EventFilter storedEventFilter,
      ErrorInfoManager errorInfoManager,
      final Function<SkyKey, Runnable> runnableMaker,
      final ForkJoinPool forkJoinPool) {
    this.graph = graph;
    this.graphVersion = graphVersion;
    this.skyFunctions = skyFunctions;
    this.reporter = reporter;
    this.replayingNestedSetEventVisitor =
        new NestedSetVisitor<>(new NestedSetEventReceiver(reporter), emittedEventState.eventState);
    this.replayingNestedSetPostableVisitor =
        new NestedSetVisitor<>(
            new NestedSetPostableReceiver(reporter), emittedEventState.postableState);
    this.keepGoing = keepGoing;
    this.progressReceiver = Preconditions.checkNotNull(progressReceiver);
    this.storedEventFilter = storedEventFilter;
    this.errorInfoManager = errorInfoManager;
    visitorSupplier =
        Suppliers.memoize(
            new Supplier<NodeEntryVisitor>() {
              @Override
              public NodeEntryVisitor get() {
                return new NodeEntryVisitor(
                    forkJoinPool, progressReceiver, runnableMaker);
              }
            });
  }

  Map<SkyKey, ? extends NodeEntry> getBatchValues(
      @Nullable SkyKey parent, Reason reason, Iterable<SkyKey> keys) throws InterruptedException {
    return graph.getBatch(parent, reason, keys);
  }

  /**
   * Signals all parents that this node is finished. If {@code enqueueParents} is true, also
   * enqueues any parents that are ready. Otherwise, this indicates that we are building this node
   * after the main build aborted, so skip any parents that are already done (that can happen with
   * cycles).
   */
  void signalValuesAndEnqueueIfReady(
      SkyKey skyKey, Iterable<SkyKey> keys, Version version, EnqueueParentBehavior enqueueParents)
      throws InterruptedException {
    // No fields of the entry are needed here, since we're just enqueuing for evaluation, but more
    // importantly, these hints are not respected for not-done nodes. If they are, we may need to
    // alter this hint.
    Map<SkyKey, ? extends NodeEntry> batch = graph.getBatch(skyKey, Reason.SIGNAL_DEP, keys);
    switch (enqueueParents) {
      case ENQUEUE:
        for (SkyKey key : keys) {
          NodeEntry entry = Preconditions.checkNotNull(batch.get(key), key);
          if (entry.signalDep(version)) {
            getVisitor().enqueueEvaluation(key);
          }
        }
        return;
      case SIGNAL:
        for (SkyKey key : keys) {
          NodeEntry entry = Preconditions.checkNotNull(batch.get(key), key);
          if (!entry.isDone()) {
            // In cycles, we can have parents that are already done.
            entry.signalDep(version);
          }
        }
        return;
      case NO_ACTION:
        return;
      default:
        throw new IllegalStateException(enqueueParents + ", " + skyKey);
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

  NestedSetVisitor<TaggedEvents> getReplayingNestedSetEventVisitor() {
    return replayingNestedSetEventVisitor;
  }

  NestedSetVisitor<Postable> getReplayingNestedSetPostableVisitor() {
    return replayingNestedSetPostableVisitor;
  }

  ExtendedEventHandler getReporter() {
    return reporter;
  }

  ImmutableMap<SkyFunctionName, ? extends SkyFunction> getSkyFunctions() {
    return skyFunctions;
  }

  EventFilter getStoredEventFilter() {
    return storedEventFilter;
  }

  ErrorInfoManager getErrorInfoManager() {
    return errorInfoManager;
  }

  /** Receives the events from the NestedSet and delegates to the reporter. */
  private static class NestedSetEventReceiver implements NestedSetVisitor.Receiver<TaggedEvents> {

    private final ExtendedEventHandler reporter;

    public NestedSetEventReceiver(ExtendedEventHandler reporter) {
      this.reporter = reporter;
    }

    @Override
    public void accept(TaggedEvents events) {
      String tag = events.getTag();
      for (Event e : events.getEvents()) {
        reporter.handle(e.withTag(tag));
      }
    }
  }

  /** Receives the postables from the NestedSet and delegates to the reporter. */
  private static class NestedSetPostableReceiver implements NestedSetVisitor.Receiver<Postable> {

    private final ExtendedEventHandler reporter;

    public NestedSetPostableReceiver(ExtendedEventHandler reporter) {
      this.reporter = reporter;
    }

    @Override
    public void accept(Postable post) {
      reporter.post(post);
    }
  }
}
