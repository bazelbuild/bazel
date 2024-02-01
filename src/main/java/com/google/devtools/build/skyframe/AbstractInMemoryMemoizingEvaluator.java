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

import static com.google.common.base.Preconditions.checkNotNull;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.profiler.AutoProfiler;
import com.google.devtools.build.lib.profiler.GoogleAutoProfilerUtils;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.io.PrintStream;
import java.time.Duration;
import java.util.Collection;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BiConsumer;
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
  protected Set<SkyKey> valuesToDelete = new LinkedHashSet<>();
  private Set<SkyKey> valuesToDirty = new LinkedHashSet<>();
  protected Map<SkyKey, Delta> valuesToInject = new HashMap<>();
  private final DeletingInvalidationState deleterState = new DeletingInvalidationState();
  protected final Differencer differencer;
  protected final GraphInconsistencyReceiver graphInconsistencyReceiver;
  protected final EventFilter eventFilter;

  /**
   * Whether to store edges in the graph. Can be false to save memory, in which case incremental
   * builds are not possible, and all evaluations will be at {@link Version#constant}.
   */
  protected final boolean keepEdges;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  protected final EmittedEventState emittedEventState;

  // Null until the first incremental evaluation completes. Always null when not keeping edges.
  @Nullable protected IntVersion lastGraphVersion = null;

  protected AbstractInMemoryMemoizingEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      DirtyAndInflightTrackingProgressReceiver progressReceiver,
      EventFilter eventFilter,
      EmittedEventState emittedEventState,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      boolean keepEdges) {
    this.skyFunctions = checkNotNull(skyFunctions);
    this.differencer = checkNotNull(differencer);
    this.progressReceiver = checkNotNull(progressReceiver);
    this.emittedEventState = checkNotNull(emittedEventState);
    this.eventFilter = checkNotNull(eventFilter);
    this.graphInconsistencyReceiver = checkNotNull(graphInconsistencyReceiver);
    this.keepEdges = keepEdges;
  }

  @Override
  public void delete(Predicate<SkyKey> deletePredicate) {
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

  @Override
  public void deleteDirty(long versionAgeLimit) {
    Preconditions.checkArgument(versionAgeLimit >= 0, versionAgeLimit);
    Version threshold = IntVersion.of(lastGraphVersion.getVal() - versionAgeLimit);
    valuesToDelete.addAll(
        Sets.filter(
            progressReceiver.getUnenqueuedDirtyKeys(),
            skyKey -> {
              NodeEntry entry = getInMemoryGraph().get(null, Reason.OTHER, skyKey);
              Preconditions.checkNotNull(entry, skyKey);
              Preconditions.checkState(entry.isDirty(), skyKey);
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
    Map<String, AtomicInteger> counter = new HashMap<>();
    for (InMemoryNodeEntry entry : getInMemoryGraph().getAllNodeEntries()) {
      String mapKey = entry.getKey().functionName().getName();
      counter.putIfAbsent(mapKey, new AtomicInteger());
      counter.get(mapKey).incrementAndGet();
    }
    for (Entry<String, AtomicInteger> entry : counter.entrySet()) {
      out.println(entry.getKey() + "\t" + entry.getValue()); // \t is spreadsheet-friendly.
    }
  }

  /**
   * @return true if finished successfully, false if thread was aborted
   */
  private boolean processValues(
      Predicate<String> filter, BiConsumer<String, InMemoryNodeEntry> consumer)
      throws InterruptedException {
    for (InMemoryNodeEntry entry : getInMemoryGraph().getAllNodeEntries()) {
      // This can be very long running on large graphs so check for user abort requests.
      if (Thread.interrupted()) {
        return false;
      }

      String canonicalizedKey = entry.getKey().getCanonicalName();
      if (!filter.test(canonicalizedKey) || !entry.isDone()) {
        continue;
      }

      consumer.accept(canonicalizedKey, entry);
    }

    return true;
  }

  @Override
  public final void dumpValues(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    boolean interrupted =
        !processValues(
            filter,
            (canonicalizedKey, entry) -> {
              out.println(canonicalizedKey);
              entry.getValue().debugPrint(out);
              out.println();
            });
    if (interrupted) {
      out.println("aborting");
      throw new InterruptedException();
    }
  }

  @Override
  public final void dumpDeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    boolean interrupted =
        !processValues(
            filter,
            (canonicalizedKey, entry) -> {
              out.println(canonicalizedKey);
              if (entry.keepsEdges()) {
                GroupedDeps deps =
                    GroupedDeps.decompress(entry.getCompressedDirectDepsForDoneEntry());
                for (int i = 0; i < deps.numGroups(); i++) {
                  out.format("  Group %d:\n", i + 1);
                  for (SkyKey dep : deps.getDepGroup(i)) {
                    out.print("    ");
                    out.println(dep.getCanonicalName());
                  }
                }
              } else {
                out.println("  (direct deps not stored)");
              }
              out.println();
            });
    if (interrupted) {
      out.println("aborting");
      throw new InterruptedException();
    }
  }

  @Override
  public final void dumpRdeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    boolean interrupted =
        !processValues(
            filter,
            (canonicalizedKey, entry) -> {
              out.println(canonicalizedKey);
              if (entry.keepsEdges()) {
                Collection<SkyKey> rdeps = entry.getReverseDepsForDoneEntry();
                for (SkyKey rdep : rdeps) {
                  out.print("    ");
                  out.println(rdep.getCanonicalName());
                }
              } else {
                out.println("  (rdeps not stored)");
              }
              out.println();
            });
    if (interrupted) {
      out.println("aborting");
      throw new InterruptedException();
    }
  }

  @Override
  public void cleanupInterningPools() {
    getInMemoryGraph().cleanupInterningPools();
  }

  protected void invalidate(Iterable<SkyKey> diff) {
    Iterables.addAll(valuesToDirty, diff);
  }

  /**
   * Removes entries in {@code valuesToInject} whose values are equal to the present values in the
   * graph.
   */
  protected void pruneInjectedValues(Map<SkyKey, Delta> valuesToInject) {
    for (Iterator<Entry<SkyKey, Delta>> it = valuesToInject.entrySet().iterator(); it.hasNext(); ) {
      Map.Entry<SkyKey, Delta> entry = it.next();
      SkyKey key = entry.getKey();
      SkyValue newValue = entry.getValue().newValue();
      NodeEntry prevEntry = getInMemoryGraph().get(null, Reason.OTHER, key);
      @Nullable Version newMtsv = entry.getValue().newMaxTransitiveSourceVersion();
      if (prevEntry != null && prevEntry.isDone()) {
        @Nullable Version oldMtsv = prevEntry.getMaxTransitiveSourceVersion();
        if (keepEdges) {
          try {
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
          } catch (InterruptedException e) {
            throw new IllegalStateException(
                "InMemoryGraph does not throw: " + entry + ", " + prevEntry, e);
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
  protected void injectValues(Version version) {
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

  protected void performInvalidation() throws InterruptedException {
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

  protected final Version getNextGraphVersion() {
    if (!keepEdges) {
      return Version.constant();
    } else if (lastGraphVersion == null) {
      return IntVersion.of(0);
    } else {
      return lastGraphVersion.next();
    }
  }
}
