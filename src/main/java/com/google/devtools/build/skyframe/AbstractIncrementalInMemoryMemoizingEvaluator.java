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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Iterables;
import com.google.devtools.build.skyframe.Differencer.DiffWithDelta.Delta;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DeletingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.DirtyingInvalidationState;
import com.google.devtools.build.skyframe.InvalidatingNodeVisitor.InvalidationState;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Partial implementation of {@link MemoizingEvaluator} with expanded support for incremental and
 * non-incremental evaluations on an {@link InMemoryGraph}.
 */
public abstract class AbstractIncrementalInMemoryMemoizingEvaluator
    extends AbstractInMemoryMemoizingEvaluator {

  protected final ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions;
  protected final DirtyTrackingProgressReceiver progressReceiver;

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
  final boolean keepEdges;

  // Values that the caller explicitly specified are assumed to be changed -- they will be
  // re-evaluated even if none of their children are changed.
  private final InvalidationState invalidatorState = new DirtyingInvalidationState();

  protected final EmittedEventState emittedEventState;

  // Null until the first incremental evaluation completes. Always null when not keeping edges.
  @Nullable protected IntVersion lastGraphVersion = null;

  protected AbstractIncrementalInMemoryMemoizingEvaluator(
      ImmutableMap<SkyFunctionName, SkyFunction> skyFunctions,
      Differencer differencer,
      DirtyTrackingProgressReceiver dirtyTrackingProgressReceiver,
      EventFilter eventFilter,
      EmittedEventState emittedEventState,
      GraphInconsistencyReceiver graphInconsistencyReceiver,
      boolean keepEdges) {
    this.skyFunctions = checkNotNull(skyFunctions);
    this.differencer = checkNotNull(differencer);
    this.progressReceiver = checkNotNull(dirtyTrackingProgressReceiver);
    this.emittedEventState = checkNotNull(emittedEventState);
    this.eventFilter = checkNotNull(eventFilter);
    this.graphInconsistencyReceiver = checkNotNull(graphInconsistencyReceiver);
    this.keepEdges = keepEdges;
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
}
