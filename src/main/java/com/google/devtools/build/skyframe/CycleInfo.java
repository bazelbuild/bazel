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

import static com.google.common.base.Preconditions.checkArgument;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Function;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicates;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;

/**
 * Data for a single cycle in the graph, together with the path to the cycle. For any value, the
 * head of path to the cycle should be the value itself, or, if the value is actually in the cycle,
 * the cycle should start with the value.
 */
public class CycleInfo {
  private final ImmutableList<SkyKey> cycle;
  private final ImmutableList<SkyKey> pathToCycle;

  @VisibleForTesting
  public CycleInfo(Iterable<SkyKey> cycle) {
    this(ImmutableList.<SkyKey>of(), cycle);
  }

  public CycleInfo(Iterable<SkyKey> pathToCycle, Iterable<SkyKey> cycle) {
    this.pathToCycle = ImmutableList.copyOf(pathToCycle);
    this.cycle = ImmutableList.copyOf(cycle);
    checkArgument(!this.cycle.isEmpty(), "Cycle cannot be empty: %s", this);
  }

  // If a cycle is already known, but we are processing a value in the middle of the cycle, we need
  // to shift the cycle so that the value is at the head.
  private CycleInfo(Iterable<SkyKey> cycle, int cycleStart) {
    Preconditions.checkState(cycleStart >= 0, cycleStart);
    ImmutableList.Builder<SkyKey> cycleTail = ImmutableList.builder();
    ImmutableList.Builder<SkyKey> cycleHead = ImmutableList.builder();
    int index = 0;
    for (SkyKey key : cycle) {
      if (index >= cycleStart) {
        cycleHead.add(key);
      } else {
        cycleTail.add(key);
      }
      index++;
    }
    Preconditions.checkState(cycleStart < index, "%s >= %s ??", cycleStart, index);
    this.cycle = cycleHead.addAll(cycleTail.build()).build();
    this.pathToCycle = ImmutableList.of();
  }

  public ImmutableList<SkyKey> getCycle() {
    return cycle;
  }

  public ImmutableList<SkyKey> getPathToCycle() {
    return pathToCycle;
  }

  public SkyKey getTopKey() {
    return pathToCycle.isEmpty() ? cycle.get(0) : pathToCycle.get(0);
  }

  // Given a cycle and a value, if the value is part of the cycle, shift the cycle. Otherwise,
  // prepend the value to the head of pathToCycle.
  private static CycleInfo normalizeCycle(final SkyKey value, CycleInfo cycle) {
    int index = cycle.cycle.indexOf(value);
    if (index > -1) {
      if (!cycle.pathToCycle.isEmpty()) {
        // The head value we are considering is already part of a cycle, but we have reached it by a
        // roundabout way. Since we should have reached it directly as well, filter this roundabout
        // way out. Example (c has a dependence on top):
        //          top
        //         /  ^
        //        a   |
        //       / \ /
        //      b-> c
        // In the traversal, we start at top, visit a, then c, then top. This yields the
        // cycle {top,a,c}. Then we visit b, getting (b, {top,a,c}). Then we construct the full
        // error for a. The error should just be the cycle {top,a,c}, but we have an extra copy of
        // it via the path through b.
        return null;
      }
      return new CycleInfo(cycle.cycle, index);
    }
    return new CycleInfo(Iterables.concat(ImmutableList.of(value), cycle.pathToCycle),
        cycle.cycle);
  }

  /**
   * Normalize multiple cycles. This includes removing multiple paths to the same cycle, so that
   * a value does not depend on the same cycle multiple ways through the same child value. Note that
   * a value can still depend on the same cycle multiple ways, it's just that each way must be
   * through a different child value (a path with a different first element).
   */
  static Iterable<CycleInfo> prepareCycles(final SkyKey value, Iterable<CycleInfo> cycles) {
    final Set<ImmutableList<SkyKey>> alreadyDoneCycles = new HashSet<>();
    return Iterables.filter(Iterables.transform(cycles,
        new Function<CycleInfo, CycleInfo>() {
          @Override
          public CycleInfo apply(CycleInfo input) {
            CycleInfo normalized = normalizeCycle(value, input);
            if (normalized != null && alreadyDoneCycles.add(normalized.cycle)) {
              return normalized;
            }
            return null;
          }
    }), Predicates.notNull());
  }

  @Override
  public int hashCode() {
    return Objects.hash(cycle, pathToCycle);
  }

  @Override
  public boolean equals(Object that) {
    if (this == that) {
      return true;
    }
    if (!(that instanceof CycleInfo)) {
      return false;
    }

    CycleInfo thatCycle = (CycleInfo) that;
    return thatCycle.cycle.equals(this.cycle) && thatCycle.pathToCycle.equals(this.pathToCycle);
  }

  @Override
  public String toString() {
    return Iterables.toString(pathToCycle) + " -> " + Iterables.toString(cycle);
  }
}
