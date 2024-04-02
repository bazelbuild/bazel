// Copyright 2023 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.collect.HashMultimap;
import com.google.common.collect.HashMultiset;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Multiset;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.skyframe.Differencer.Diff;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.errorprone.annotations.ForOverride;
import java.io.PrintStream;
import java.time.Duration;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.function.Consumer;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/**
 * Partial implementation of {@link MemoizingEvaluator} with support for incremental and
 * non-incremental evaluations on an {@link InMemoryGraph}.
 */
public abstract class AbstractInMemoryMemoizingEvaluator implements MemoizingEvaluator {
  private static final Duration MIN_TIME_TO_LOG_DELETION = Duration.ofMillis(10);

  protected final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  protected final DirtyAndInflightTrackingProgressReceiver progressReceiver;

  // State related to invalidation and deletion.
  private Set<SkyKey> valuesToDelete = new LinkedHashSet<>();
  private Set<SkyKey> valuesToDirty = new LinkedHashSet<>();
  private Map<SkyKey, Delta> valuesToInject = new HashMap<>();
  private final DeletingInvalidationState deleterState = new DeletingInvalidationState();
  private final Differencer differencer;
  protected final GraphInconsistencyReceiver graphInconsistencyReceiver;
  private final EventFilter eventFilter;

  /**
   * Whether to store edges in the graph. Can be false to save memory, in which case incremental
   * builds are not possible, and all evaluations will be at {@link Version#constant}.
   */
  protected final boolean keepEdges;

  private final Version minimalVersion;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  private final EmittedEventState emittedEventState;

  // Null until the first incremental evaluation completes. Always null when not keeping edges.
  @Nullable private IntVersion lastGraphVersion = null;

  private final AtomicBoolean evaluating = new AtomicBoolean(false);

  private Set<SkyKey> latestTopLevelEvaluations = new HashSet<>();

  private boolean skyfocusEnabled;

  protected AbstractInMemoryMemoizingEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      DirtyAndInflightTrackingProgressReceiver progressReceiver,
      EventFilter eventFilter,
      EmittedEventState emittedEventState,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      boolean keepEdges,
      Version minimalVersion) {
    this.skyFunctions = checkNotNull(skyFunctions);
    this.differencer = checkNotNull(differencer);
    this.progressReceiver = checkNotNull(progressReceiver);
    this.emittedEventState = checkNotNull(emittedEventState);
    this.eventFilter = checkNotNull(eventFilter);
    this.graphInconsistencyReceiver = checkNotNull(graphInconsistencyReceiver);
    this.keepEdges = keepEdges;
    this.minimalVersion = checkNotNull(minimalVersion);
  }

  @Override
  public <T extends SkyValue> EvaluationResult<T> evaluate(
      Iterable<? extends SkyKey> roots, EvaluationContext evaluationContext)
      throws InterruptedException {
    // NOTE: Performance critical code. See bug "Null build performance parity".
    Version graphVersion = getNextGraphVersion();
    setAndCheckEvaluateState(true, roots);

    // Only remember roots for Skyfocus if we're tracking incremental states by keeping edges.
    if (keepEdges && skyfocusEnabled) {
      // Remember the top level evaluation of the build invocation for post-build consumption.
      Iterables.addAll(latestTopLevelEvaluations, roots);
    }

    // Mark for removal any nodes from the previous evaluation that were still inflight or were
    // rewound but did not complete successfully. When the invalidator runs, it will delete the
    // reverse transitive closure.
    valuesToDelete.addAll(progressReceiver.getAndClearInflightKeys());
    valuesToDelete.addAll(progressReceiver.getAndClearUnsuccessfullyRewoundKeys());
    try {
      // The RecordingDifferencer implementation is not quite working as it should be at this point.
      // It clears the internal data structures after getDiff is called and will not return
      // diffs for historical versions. This makes the following code sensitive to interrupts.
      // Ideally we would simply not update lastGraphVersion if an interrupt occurs.
      Diff diff =
          differencer.getDiff(
              new DelegatingWalkableGraph(getInMemoryGraph()), lastGraphVersion, graphVersion);
      if (!diff.isEmpty() || !valuesToInject.isEmpty() || !valuesToDelete.isEmpty()) {
        valuesToInject.putAll(diff.changedKeysWithNewValues());
        invalidate(diff.changedKeysWithoutNewValues());
        pruneInjectedValues(valuesToInject);
        invalidate(valuesToInject.keySet());

        performInvalidation();
        injectValues(graphVersion);
      }
      ProcessableGraph graph = getGraphForEvaluation(evaluationContext);

      EvaluationResult<T> result;
      try (SilentCloseable c = Profiler.instance().profile("ParallelEvaluator.eval")) {
        ParallelEvaluator evaluator =
            new ParallelEvaluator(
                graph,
                graphVersion,
                minimalVersion,
                skyFunctions,
                evaluationContext.getEventHandler(),
                emittedEventState,
                eventFilter,
                ErrorInfoManager.UseChildErrorInfoIfNecessary.INSTANCE,
                evaluationContext.getKeepGoing(),
                progressReceiver,
                graphInconsistencyReceiver,
                evaluationContext
                    .getExecutor()
                    .orElseGet(
                        () ->
                            AbstractQueueVisitor.create(
                                "skyframe-evaluator-memoizing",
                                evaluationContext.getParallelism(),
                                ParallelEvaluatorErrorClassifier.instance())),
                new SimpleCycleDetector(),
                evaluationContext.getUnnecessaryTemporaryStateDropperReceiver());
        result = evaluator.eval(roots);
      }
      return EvaluationResult.<T>builder()
          .mergeFrom(result)
          .setWalkableGraph(new DelegatingWalkableGraph(getInMemoryGraph()))
          .build();
    } finally {
      if (keepEdges) {
        lastGraphVersion = (IntVersion) graphVersion;
      }
      setAndCheckEvaluateState(false, roots);
    }
  }

  @ForOverride
  protected ProcessableGraph getGraphForEvaluation(EvaluationContext evaluationContext)
      throws InterruptedException {
    return getInMemoryGraph();
  }

  @Override
  public final void delete(Predicate<SkyKey> deletePredicate) {
    try (AutoProfiler ignored =
        GoogleAutoProfilerUtils.logged("deletion marking", MIN_TIME_TO_LOG_DELETION)) {
      Set<SkyKey> toDelete = Sets.newConcurrentHashSet();
      getInMemoryGraph()
          .parallelForEach(
              e -> {
                if (e.isDirty() || deletePredicate.test(e.getKey())) {
                  toDelete.add(e.getKey());
                }
              });
      valuesToDelete.addAll(toDelete);
    }
  }

  private void setAndCheckEvaluateState(boolean newValue, Iterable<? extends SkyKey> roots) {
    checkState(
        evaluating.getAndSet(newValue) != newValue, "Re-entrant evaluation for request: %s", roots);
  }

  @Override
  public boolean getSkyfocusEnabled() {
    return skyfocusEnabled;
  }

  @Override
  public void setSkyfocusEnabled(boolean enabled) {
    this.skyfocusEnabled = enabled;
  }

  @Override
  public boolean skyfocusSupported() {
    return true;
  }

  @Override
  public final void deleteDirty(long versionAgeLimit) {
    checkArgument(versionAgeLimit >= 0, versionAgeLimit);
    Version threshold = IntVersion.of(lastGraphVersion.getVal() - versionAgeLimit);
    valuesToDelete.addAll(
        Sets.filter(
            progressReceiver.getUnenqueuedDirtyKeys(),
            skyKey -> {
              NodeEntry entry = checkNotNull(getInMemoryGraph().getIfPresent(skyKey), skyKey);
              checkState(entry.isDirty(), skyKey);
              return entry.getVersion().atMost(threshold);
            }));
  }

  @Override
  public final Map<SkyKey, SkyValue> getValues() {
    return getInMemoryGraph().getValues();
  }

  @Override
  public final Map<SkyKey, SkyValue> getDoneValues() {
    return getInMemoryGraph().getDoneValues();
  }

  private static boolean isDone(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }

  @Override
  @Nullable
  public final SkyValue getExistingValue(SkyKey key) {
    InMemoryNodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    return isDone(entry) ? entry.getValue() : null;
  }

  @Override
  @Nullable
  public final ErrorInfo getExistingErrorForTesting(SkyKey key) {
    InMemoryNodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    return isDone(entry) ? entry.getErrorInfo() : null;
  }

  @Nullable
  @Override
  public final InMemoryNodeEntry getExistingEntryAtCurrentlyEvaluatingVersion(SkyKey key) {
    return getInMemoryGraph().getIfPresent(key);
  }

  @Override
  public final void dumpSummary(PrintStream out) {
    long nodes = 0;
    long edges = 0;
    for (InMemoryNodeEntry entry : getInMemoryGraph().getAllNodeEntries()) {
      nodes++;
      if (entry.isDone() && entry.keepsEdges()) {
        edges += Iterables.size(entry.getDirectDeps());
      }
    }
    out.println("Node count: " + nodes);
    out.println("Edge count: " + edges);
  }

  @Override
  public final void dumpCount(PrintStream out) {
    Multiset<SkyFunctionName> counter = HashMultiset.create();
    for (InMemoryNodeEntry entry : getInMemoryGraph().getAllNodeEntries()) {
      counter.add(entry.getKey().functionName());
    }
    for (Multiset.Entry<SkyFunctionName> entry : counter.entrySet()) {
      out.println(entry.getElement() + "\t" + entry.getCount()); // \t is spreadsheet-friendly.
    }
  }

  private void processGraphForDumpCommand(
      Predicate<String> filter, PrintStream out, Consumer<InMemoryNodeEntry> consumer)
      throws InterruptedException {
    for (InMemoryNodeEntry entry : getInMemoryGraph().getAllNodeEntries()) {
      // This can be very long running on large graphs so check for user abort requests.
      if (Thread.interrupted()) {
        out.println("aborting");
        throw new InterruptedException();
      }

      if (!filter.test(entry.getKey().getCanonicalName()) || !entry.isDone()) {
        continue;
      }

      consumer.accept(entry);
    }
  }

  @Override
  public final void dumpValues(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    processGraphForDumpCommand(
        filter,
        out,
        entry -> {
          out.println(entry.getKey().getCanonicalName());
          entry.getValue().debugPrint(out);
          out.println();
        });
  }

  @Override
  public final void dumpDeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    processGraphForDumpCommand(
        filter,
        out,
        entry -> {
          String canonicalizedKey = entry.getKey().getCanonicalName();
          out.println(canonicalizedKey);
          if (entry.keepsEdges()) {
            GroupedDeps deps = GroupedDeps.decompress(entry.getCompressedDirectDepsForDoneEntry());
            for (int i = 0; i < deps.numGroups(); i++) {
              out.format("  Group %d:\n", i + 1);
              for (SkyKey dep : deps.getDepGroup(i)) {
                out.print("    ");
                out.println(dep.getCanonicalName());
                out.println(); // newline for readability
              }
            }
          } else {
            out.println("  (direct deps not stored)");
          }
          out.println();
        });
  }

  @Override
  public final void dumpFunctionGraph(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    HashMultimap<SkyFunctionName, SkyFunctionName> seen = HashMultimap.create();
    out.println("digraph {");
    processGraphForDumpCommand(
        filter,
        out,
        entry -> {
          if (entry.keepsEdges()) {
            SkyFunctionName source = entry.getKey().functionName();
            for (SkyKey dep : entry.getDirectDeps()) {
              SkyFunctionName dest = dep.functionName();
              if (!seen.put(source, dest)) {
                continue;
              }
              out.format("  \"%s\" -> \"%s\"\n", source, dest);
            }
          }
        });
    out.println("}");
  }

  @Override
  public final void dumpRdeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    processGraphForDumpCommand(
        filter,
        out,
        entry -> {
          out.println(entry.getKey().getCanonicalName());
          if (entry.keepsEdges()) {
            Collection<SkyKey> rdeps = entry.getReverseDepsForDoneEntry();
            for (SkyKey rdep : rdeps) {
              out.print("    ");
              out.println(rdep.getCanonicalName());
              out.println();
            }
          } else {
            out.println("  (rdeps not stored)");
          }
          out.println();
        });
  }

  @Override
  public final void cleanupInterningPools() {
    getInMemoryGraph().cleanupInterningPools();
  }

  private void invalidate(Iterable<SkyKey> diff) {
    Iterables.addAll(valuesToDirty, diff);
  }

  /**
   * Removes entries in {@code valuesToInject} whose values are equal to the present values in the
   * graph.
   */
  private void pruneInjectedValues(Map<SkyKey, Delta> valuesToInject) {
    for (Iterator<Entry<SkyKey, Delta>> it = valuesToInject.entrySet().iterator(); it.hasNext(); ) {
      Map.Entry<SkyKey, Delta> entry = it.next();
      SkyKey key = entry.getKey();
      SkyValue newValue = entry.getValue().newValue();
      InMemoryNodeEntry prevEntry = getInMemoryGraph().getIfPresent(key);
      @Nullable Version newMtsv = entry.getValue().newMaxTransitiveSourceVersion();
      if (isDone(prevEntry)) {
        @Nullable Version oldMtsv = prevEntry.getMaxTransitiveSourceVersion();
        if (keepEdges) {
          if (!prevEntry.hasAtLeastOneDep()) {
            if (newValue.equals(prevEntry.getValue())
                && !valuesToDirty.contains(key)
                && !valuesToDelete.contains(key)
                && Objects.equals(newMtsv, oldMtsv)) {
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
        } else {
          // No incrementality. Just delete the old value from the graph. The new value is about to
          // be injected.
          getInMemoryGraph().remove(key);
        }
      }
    }
  }

  /** Injects values in {@code valuesToInject} into the graph. */
  private void injectValues(Version version) {
    if (valuesToInject.isEmpty()) {
      return;
    }
    try {
      ParallelEvaluator.injectValues(valuesToInject, version, getInMemoryGraph(), progressReceiver);
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph doesn't throw interrupts", e);
    }
    // Start with a new map to avoid bloat since clear() does not downsize the map.
    valuesToInject = new HashMap<>();
  }

  private void performInvalidation() throws InterruptedException {
    EagerInvalidator.delete(
        getInMemoryGraph(), valuesToDelete, progressReceiver, deleterState, keepEdges);
    // Note that clearing the valuesToDelete would not do an internal resizing. Therefore, if any
    // build has a large set of dirty values, subsequent operations (even clearing) will be slower.
    // Instead, just start afresh with a new LinkedHashSet.
    valuesToDelete = new LinkedHashSet<>();

    EagerInvalidator.invalidate(
        getInMemoryGraph(), valuesToDirty, progressReceiver, invalidatorState);
    // Ditto.
    valuesToDirty = new LinkedHashSet<>();
  }

  private Version getNextGraphVersion() {
    if (!keepEdges) {
      return Version.constant();
    } else if (lastGraphVersion == null) {
      return IntVersion.of(0);
    } else {
      return lastGraphVersion.next();
    }
  }

  public Set<SkyKey> getLatestTopLevelEvaluations() {
    return latestTopLevelEvaluations;
  }

  @Override
  public void cleanupLatestTopLevelEvaluations() {
    latestTopLevelEvaluations = new HashSet<>();
  }
}
