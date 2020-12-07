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
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * An inmemory implementation that uses the eager invalidation strategy. This class is, by itself,
 * not thread-safe. Neither is it thread-safe to use this class in parallel with any of the
 * returned graphs. However, it is allowed to access the graph from multiple threads as long as
 * that does not happen in parallel with an {@link #evaluate} call.
 *
 * <p>This memoizing evaluator requires a sequential versioning scheme. Evaluations
 * must pass in a monotonically increasing {@link IntVersion}.
 */
public final class InMemoryMemoizingEvaluator implements MemoizingEvaluator {

  private final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  private final DirtyTrackingProgressReceiver progressReceiver;
  // Not final only for testing.
  private InMemoryGraph graph;
  private IntVersion lastGraphVersion = null;

  // State related to invalidation and deletion.
  private Set<SkyKey> valuesToDelete = new LinkedHashSet<>();
  private Set<SkyKey> valuesToDirty = new LinkedHashSet<>();
  private Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
  private final InvalidationState deleterState = new DeletingInvalidationState();
  private final Differencer differencer;
  private final GraphInconsistencyReceiver graphInconsistencyReceiver;
  private final EventFilter eventFilter;

  // Keep edges in graph. Can be false to save memory, in which case incremental builds are
  // not possible.
  private final boolean keepEdges;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  private final EmittedEventState emittedEventState;

  private final AtomicBoolean evaluating = new AtomicBoolean(false);

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions, Differencer differencer) {
    this(skyFunctions, differencer, null);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver progressReceiver) {
    this(
        skyFunctions,
        differencer,
        progressReceiver,
        GraphInconsistencyReceiver.THROWING,
        DEFAULT_STORED_EVENT_FILTER,
        new EmittedEventState(),
        true);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver progressReceiver,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      EventFilter eventFilter,
      EmittedEventState emittedEventState,
      boolean keepEdges) {
    this.skyFunctions = ImmutableMap.copyOf(skyFunctions);
    this.differencer = Preconditions.checkNotNull(differencer);
    this.progressReceiver = new DirtyTrackingProgressReceiver(progressReceiver);
    this.graphInconsistencyReceiver = Preconditions.checkNotNull(graphInconsistencyReceiver);
    this.eventFilter = eventFilter;
    this.graph = new InMemoryGraphImpl(keepEdges);
    this.emittedEventState = emittedEventState;
    this.keepEdges = keepEdges;
  }

  private void invalidate(Iterable<SkyKey> diff) {
    Iterables.addAll(valuesToDirty, diff);
  }

  @Override
  public void delete(Predicate<SkyKey> deletePredicate) {
    valuesToDelete.addAll(
        Maps.filterEntries(
                graph.getAllValues(),
                input -> {
                  Preconditions.checkNotNull(input.getKey(), "Null SkyKey in entry: %s", input);
                  return input.getValue().isDirty() || deletePredicate.test(input.getKey());
                })
            .keySet());
  }

  @Override
  public void deleteDirty(long versionAgeLimit) {
    Preconditions.checkArgument(versionAgeLimit >= 0);
    final Version threshold = IntVersion.of(lastGraphVersion.getVal() - versionAgeLimit);
    valuesToDelete.addAll(
        Sets.filter(
            progressReceiver.getUnenqueuedDirtyKeys(),
            skyKey -> {
              NodeEntry entry = graph.get(null, Reason.OTHER, skyKey);
              Preconditions.checkNotNull(entry, skyKey);
              Preconditions.checkState(entry.isDirty(), skyKey);
              return entry.getVersion().atMost(threshold);
            }));
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots, Version version, EvaluationContext evaluationContext)
      throws InterruptedException {
    // NOTE: Performance critical code. See bug "Null build performance parity".
    IntVersion intVersion = (IntVersion) version;
    Preconditions.checkState((lastGraphVersion == null && intVersion.getVal() == 0)
        || version.equals(lastGraphVersion.next()),
        "InMemoryGraph supports only monotonically increasing Integer versions: %s %s",
        lastGraphVersion, version);
    setAndCheckEvaluateState(true, roots);
    try {
      // Mark for removal any inflight nodes from the previous evaluation.
      valuesToDelete.addAll(progressReceiver.getAndClearInflightKeys());

      // The RecordingDifferencer implementation is not quite working as it should be at this point.
      // It clears the internal data structures after getDiff is called and will not return
      // diffs for historical versions. This makes the following code sensitive to interrupts.
      // Ideally we would simply not update lastGraphVersion if an interrupt occurs.
      Diff diff =
          differencer.getDiff(new DelegatingWalkableGraph(graph), lastGraphVersion, version);
      if (!diff.isEmpty() || !valuesToInject.isEmpty() || !valuesToDelete.isEmpty()) {
        valuesToInject.putAll(diff.changedKeysWithNewValues());
        invalidate(diff.changedKeysWithoutNewValues());
        pruneInjectedValues(valuesToInject);
        invalidate(valuesToInject.keySet());

        performInvalidation();
        injectValues(intVersion);
      }

      EvaluationResult<T> result;
      try (SilentCloseable c = Profiler.instance().profile("ParallelEvaluator.eval")) {
        ParallelEvaluator evaluator =
            new ParallelEvaluator(
                graph,
                version,
                skyFunctions,
                evaluationContext.getEventHandler(),
                emittedEventState,
                eventFilter,
                ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
                evaluationContext.getKeepGoing(),
                progressReceiver,
                graphInconsistencyReceiver,
                evaluationContext
                    .getExecutorServiceSupplier()
                    .orElse(
                        () ->
                            AbstractQueueVisitor.createExecutorService(
                                evaluationContext.getParallelism(), "skyframe-evaluator")),
                new SimpleCycleDetector(),
                EvaluationVersionBehavior.GRAPH_VERSION);
        result = evaluator.eval(roots);
      }
      return EvaluationResult.<T>builder()
          .mergeFrom(result)
          .setWalkableGraph(new DelegatingWalkableGraph(graph))
          .build();
    } finally {
      lastGraphVersion = intVersion;
      setAndCheckEvaluateState(false, roots);
    }
  }

  /**
   * Removes entries in {@code valuesToInject} whose values are equal to the present values in the
   * graph.
   */
  private void pruneInjectedValues(Map<SkyKey, SkyValue> valuesToInject) {
    for (Iterator<Map.Entry<SkyKey, SkyValue>> it = valuesToInject.entrySet().iterator();
        it.hasNext(); ) {
      Map.Entry<SkyKey, SkyValue> entry = it.next();
      SkyKey key = entry.getKey();
      SkyValue newValue = entry.getValue();
      NodeEntry prevEntry = graph.get(null, Reason.OTHER, key);
      if (prevEntry != null && prevEntry.isDone()) {
        if (keepEdges) {
          try {
            if (!prevEntry.hasAtLeastOneDep()) {
              if (newValue.equals(prevEntry.getValue())
                  && !valuesToDirty.contains(key)
                  && !valuesToDelete.contains(key)) {
                it.remove();
              }
            } else {
              // Rare situation of an injected dep that depends on another node. Usually the dep is
              // the error transience node. When working with external repositories, it can also be
              // an external workspace file. Don't bother injecting it, just invalidate it.
              // We'll wastefully evaluate the node freshly during evaluation, but this happens very
              // rarely.
              valuesToDirty.add(key);
              it.remove();
            }
          } catch (InterruptedException e) {
            throw new IllegalStateException(
                "InMemoryGraph does not throw: " + entry + ", " + prevEntry, e);
          }
        } else {
          // No incrementality. Just delete the old value from the graph. The new value is about to
          // be injected.
          graph.remove(key);
        }
      }
    }
  }

  /**
   * Injects values in {@code valuesToInject} into the graph.
   */
  private void injectValues(IntVersion version) {
    if (valuesToInject.isEmpty()) {
      return;
    }
    try {
      ParallelEvaluator.injectValues(valuesToInject, version, graph, progressReceiver);
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph doesn't throw interrupts", e);
    }
    // Start with a new map to avoid bloat since clear() does not downsize the map.
    valuesToInject = new HashMap<>();
  }

  private void performInvalidation() throws InterruptedException {
    EagerInvalidator.delete(graph, valuesToDelete, progressReceiver, deleterState, keepEdges);
    // Note that clearing the valuesToDelete would not do an internal resizing. Therefore, if any
    // build has a large set of dirty values, subsequent operations (even clearing) will be slower.
    // Instead, just start afresh with a new LinkedHashSet.
    valuesToDelete = new LinkedHashSet<>();

    EagerInvalidator.invalidate(graph, valuesToDirty, progressReceiver, invalidatorState);
    // Ditto.
    valuesToDirty = new LinkedHashSet<>();
  }

  private void setAndCheckEvaluateState(boolean newValue, Object requestInfo) {
    Preconditions.checkState(evaluating.getAndSet(newValue) != newValue,
        "Re-entrant evaluation for request: %s", requestInfo);
  }

  @Override
  public Map<SkyKey, SkyValue> getValues() {
    return graph.getValues();
  }

  @Override
  public Iterable<? extends Map.Entry<SkyKey, ? extends NodeEntry>> getGraphEntries() {
    return graph.getAllValuesMutable().entrySet();
  }

  @Override
  public Map<SkyKey, SkyValue> getDoneValues() {
    return graph.getDoneValues();
  }

  private static boolean isDone(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }

  @Override
  @Nullable
  public SkyValue getExistingValue(SkyKey key) {
    NodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    try {
      return isDone(entry) ? entry.getValue() : null;
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph does not throw" + key + ", " + entry, e);
    }
  }

  @Override
  @Nullable public ErrorInfo getExistingErrorForTesting(SkyKey key) {
    NodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    try {
      return isDone(entry) ? entry.getErrorInfo() : null;
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph does not throw" + key + ", " + entry, e);
    }
  }

  @Nullable
  @Override
  public NodeEntry getExistingEntryAtCurrentlyEvaluatingVersion(SkyKey key) {
    return graph.get(null, Reason.OTHER, key);
  }

  @Override
  public void injectGraphTransformerForTesting(GraphTransformerForTesting transformer) {
    this.graph = transformer.transform(this.graph);
  }

  public ProcessableGraph getGraphForTesting() {
    return graph;
  }

  @Override
  public void dump(boolean summarize, PrintStream out) {
    if (summarize) {
      long nodes = 0;
      long edges = 0;
      for (NodeEntry entry : graph.getAllValues().values()) {
        nodes++;
        if (entry.isDone()) {
          try {
            edges += Iterables.size(entry.getDirectDeps());
          } catch (InterruptedException e) {
            throw new IllegalStateException("InMemoryGraph doesn't throw: " + entry, e);
          }
        }
      }
      out.println("Node count: " + nodes);
      out.println("Edge count: " + edges);
    } else {
      Function<SkyKey, String> keyFormatter =
          key ->
              String.format(
                  "%s:%s", key.functionName(), key.argument().toString().replace('\n', '_'));

      for (Map.Entry<SkyKey, ? extends NodeEntry> mapPair : graph.getAllValues().entrySet()) {
        SkyKey key = mapPair.getKey();
        NodeEntry entry = mapPair.getValue();
        if (entry.isDone()) {
          out.print(keyFormatter.apply(key));
          out.print("|");
          if (((InMemoryNodeEntry) entry).keepEdges() == NodeEntry.KeepEdgesPolicy.NONE) {
            out.println(" (direct deps not stored)");
          } else {
            try {
              out.println(
                  Joiner.on('|').join(Iterables.transform(entry.getDirectDeps(), keyFormatter)));
            } catch (InterruptedException e) {
              throw new IllegalStateException("InMemoryGraph doesn't throw: " + entry, e);
            }
          }
        }
      }
    }
  }

  public ImmutableMap<SkyFunctionName, SkyFunction> getSkyFunctionsForTesting() {
    return skyFunctions;
  }

  public static final EventFilter DEFAULT_STORED_EVENT_FILTER =
      new EventFilter() {
        @Override
        public boolean apply(Event event) {
          switch (event.getKind()) {
            case INFO:
            case PROGRESS:
              return false;
            default:
              return true;
          }
        }

        @Override
        public boolean storeEventsAndPosts() {
          return true;
        }
      };

  public static final EvaluatorSupplier SUPPLIER = InMemoryMemoizingEvaluator::new;
}
