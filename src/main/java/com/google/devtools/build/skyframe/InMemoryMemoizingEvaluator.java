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
import com.google.common.base.Predicate;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Maps;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.util.Preconditions;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.ParallelEvaluator.EventFilter;
import com.google.devtools.build.skyframe.ParallelEvaluator.Receiver;

import java.io.PrintStream;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

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

  private final ImmutableMap<SkyFunctionName, ? extends SkyFunction> skyFunctions;
  @Nullable private final EvaluationProgressReceiver progressReceiver;
  // Not final only for testing.
  private InMemoryGraph graph;
  private IntVersion lastGraphVersion = null;

  // State related to invalidation and deletion.
  private Set<SkyKey> valuesToDelete = new LinkedHashSet<>();
  private Set<SkyKey> valuesToDirty = new LinkedHashSet<>();
  private Map<SkyKey, SkyValue> valuesToInject = new HashMap<>();
  private final DirtyKeyTracker dirtyKeyTracker = new DirtyKeyTrackerImpl();
  private final InvalidationState deleterState = new DeletingInvalidationState();
  private final Differencer differencer;

  // Keep edges in graph. Can be false to save memory, in which case incremental builds are
  // not possible.
  private final boolean keepEdges;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  private final EmittedEventState emittedEventState;

  private final AtomicBoolean evaluating = new AtomicBoolean(false);

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, ? extends SkyFunction> skyFunctions, Differencer differencer) {
    this(skyFunctions, differencer, null);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver invalidationReceiver) {
    this(skyFunctions, differencer, invalidationReceiver, new EmittedEventState(), true);
  }

  public InMemoryMemoizingEvaluator(
      Map<SkyFunctionName, ? extends SkyFunction> skyFunctions,
      Differencer differencer,
      @Nullable EvaluationProgressReceiver invalidationReceiver,
      EmittedEventState emittedEventState,
      boolean keepEdges) {
    this.skyFunctions = ImmutableMap.copyOf(skyFunctions);
    this.differencer = Preconditions.checkNotNull(differencer);
    this.progressReceiver = invalidationReceiver;
    this.graph = new InMemoryGraph(keepEdges);
    this.emittedEventState = emittedEventState;
    this.keepEdges = keepEdges;
  }

  private void invalidate(Iterable<SkyKey> diff) {
    Iterables.addAll(valuesToDirty, diff);
  }

  @Override
  public void delete(final Predicate<SkyKey> deletePredicate) {
    valuesToDelete.addAll(
        Maps.filterEntries(graph.getAllValues(), new Predicate<Entry<SkyKey, NodeEntry>>() {
          @Override
          public boolean apply(Entry<SkyKey, NodeEntry> input) {
            Preconditions.checkNotNull(input.getKey(), "Null SkyKey in entry: %s", input);
            return input.getValue().isDirty() || deletePredicate.apply(input.getKey());
          }
        }).keySet());
  }

  @Override
  public void deleteDirty(long versionAgeLimit) {
    Preconditions.checkArgument(versionAgeLimit >= 0);
    final Version threshold = IntVersion.of(lastGraphVersion.getVal() - versionAgeLimit);
    valuesToDelete.addAll(
        Sets.filter(dirtyKeyTracker.getDirtyKeys(), new Predicate<SkyKey>() {
          @Override
          public boolean apply(SkyKey skyKey) {
            NodeEntry entry = graph.get(skyKey);
            Preconditions.checkNotNull(entry, skyKey);
            Preconditions.checkState(entry.isDirty(), skyKey);
            return entry.getVersion().atMost(threshold);
          }
        }));
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(Iterable<SkyKey> roots, Version version,
          boolean keepGoing, int numThreads, EventHandler eventHandler)
      throws InterruptedException {
    // NOTE: Performance critical code. See bug "Null build performance parity".
    IntVersion intVersion = (IntVersion) version;
    Preconditions.checkState((lastGraphVersion == null && intVersion.getVal() == 0)
        || version.equals(lastGraphVersion.next()),
        "InMemoryGraph supports only monotonically increasing Integer versions: %s %s",
        lastGraphVersion, version);
    setAndCheckEvaluateState(true, roots);
    try {
      // The RecordingDifferencer implementation is not quite working as it should be at this point.
      // It clears the internal data structures after getDiff is called and will not return
      // diffs for historical versions. This makes the following code sensitive to interrupts.
      // Ideally we would simply not update lastGraphVersion if an interrupt occurs.
      Diff diff = differencer.getDiff(new DelegatingWalkableGraph(graph), lastGraphVersion,
          version);
      valuesToInject.putAll(diff.changedKeysWithNewValues());
      invalidate(diff.changedKeysWithoutNewValues());
      pruneInjectedValues(valuesToInject);
      invalidate(valuesToInject.keySet());

      performInvalidation();
      injectValues(intVersion);

      // We must delete all nodes that are still in-flight at the end of the evaluation (in case the
      // evaluation is aborted for some reason). In order to quickly return control to the caller,
      // we store the set of such nodes for deletion at the start of the next evaluation.
      Receiver<Collection<SkyKey>> lazyDeletingReceiver =
          new Receiver<Collection<SkyKey>>() {
            @Override
            public void accept(Collection<SkyKey> skyKeys) {
              valuesToDelete.addAll(skyKeys);
            }
          };
      ParallelEvaluator evaluator =
          new ParallelEvaluator(
              graph,
              intVersion,
              skyFunctions,
              eventHandler,
              emittedEventState,
              DEFAULT_STORED_EVENT_FILTER,
              keepGoing,
              numThreads,
              progressReceiver,
              dirtyKeyTracker,
              lazyDeletingReceiver);
      EvaluationResult<T> result = evaluator.eval(roots);
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
    for (Iterator<Entry<SkyKey, SkyValue>> it = valuesToInject.entrySet().iterator();
        it.hasNext();) {
      Entry<SkyKey, SkyValue> entry = it.next();
      SkyKey key = entry.getKey();
      SkyValue newValue = entry.getValue();
      NodeEntry prevEntry = graph.get(key);
      if (prevEntry != null && prevEntry.isDone()) {
        Iterable<SkyKey> directDeps = prevEntry.getDirectDeps();
        Preconditions.checkState(Iterables.isEmpty(directDeps),
            "existing entry for %s has deps: %s", key, directDeps);
        if (newValue.equals(prevEntry.getValue())
            && !valuesToDirty.contains(key) && !valuesToDelete.contains(key)) {
          it.remove();
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
    ParallelEvaluator.injectValues(valuesToInject, version, graph, dirtyKeyTracker);
    // Start with a new map to avoid bloat since clear() does not downsize the map.
    valuesToInject = new HashMap<>();
  }

  private void performInvalidation() throws InterruptedException {
    EagerInvalidator.delete(graph, valuesToDelete, progressReceiver, deleterState, keepEdges,
        dirtyKeyTracker);
    // Note that clearing the valuesToDelete would not do an internal resizing. Therefore, if any
    // build has a large set of dirty values, subsequent operations (even clearing) will be slower.
    // Instead, just start afresh with a new LinkedHashSet.
    valuesToDelete = new LinkedHashSet<>();

    EagerInvalidator.invalidate(graph, valuesToDirty, progressReceiver, invalidatorState,
        dirtyKeyTracker);
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
  public Map<SkyKey, SkyValue> getDoneValues() {
    return graph.getDoneValues();
  }

  @Override
  @Nullable public SkyValue getExistingValueForTesting(SkyKey key) {
    return graph.getValue(key);
  }

  @Override
  @Nullable public ErrorInfo getExistingErrorForTesting(SkyKey key) {
    NodeEntry entry = graph.get(key);
    return (entry == null || !entry.isDone()) ? null : entry.getErrorInfo();
  }

  public void setGraphForTesting(InMemoryGraph graph) {
    this.graph = graph;
  }

  @Override
  public void dump(boolean summarize, PrintStream out) {
    if (summarize) {
      long nodes = 0;
      long edges = 0;
      for (NodeEntry entry : graph.getAllValues().values()) {
        nodes++;
        if (entry.isDone()) {
          edges += Iterables.size(entry.getDirectDeps());
        }
      }
      out.println("Node count: " + nodes);
      out.println("Edge count: " + edges);
    } else {
      Function<SkyKey, String> keyFormatter =
          new Function<SkyKey, String>() {
            @Override
            public String apply(SkyKey key) {
              return String.format("%s:%s",
                  key.functionName(), key.argument().toString().replace('\n', '_'));
            }
          };

      for (Entry<SkyKey, NodeEntry> mapPair : graph.getAllValues().entrySet()) {
        SkyKey key = mapPair.getKey();
        NodeEntry entry = mapPair.getValue();
        if (entry.isDone()) {
          out.print(keyFormatter.apply(key));
          out.print("|");
          out.println(Joiner.on('|').join(
              Iterables.transform(entry.getDirectDeps(), keyFormatter)));
        }
      }
    }
  }

  public ImmutableMap<SkyFunctionName, ? extends SkyFunction> getSkyFunctionsForTesting() {
    return skyFunctions;
  }
  public static final EventFilter DEFAULT_STORED_EVENT_FILTER =
      new EventFilter() {
        @Override
        public boolean apply(Event event) {
          switch (event.getKind()) {
            case INFO:
              throw new UnsupportedOperationException(
                  "SkyFunctions should not display INFO messages: "
                      + event.getLocation()
                      + ": "
                      + event.getMessage());
            case PROGRESS:
              return false;
            default:
              return true;
          }
        }

        @Override
        public boolean storeEvents() {
          return true;
        }
      };

  public static final EvaluatorSupplier SUPPLIER =
      new EvaluatorSupplier() {
        @Override
        public MemoizingEvaluator create(
            Map<SkyFunctionName, ? extends SkyFunction> skyFunctions,
            Differencer differencer,
            @Nullable EvaluationProgressReceiver invalidationReceiver,
            EmittedEventState emittedEventState,
            boolean keepEdges) {
          return new InMemoryMemoizingEvaluator(
              skyFunctions, differencer, invalidationReceiver, emittedEventState, keepEdges);
        }
      };
}
