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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import java.util.HashSet;
import java.util.Objects;
import java.util.Set;
import javax.annotation.Nullable;

/**
 * Data for a single cycle in the graph, together with the path to the cycle. For any value, the
 * head of path to the cycle should be the value itself, or, if the value is actually in the cycle,
 * the cycle should start with the value.
 */
public abstract class CycleInfo {
  public abstract ImmutableList<SkyKey> getCycle();

  public abstract ImmutableList<SkyKey> getPathToCycle();

  public abstract SkyKey getTopKey();

  public abstract boolean hasCycleDetails();

  @VisibleForTesting
  public static CycleInfo createCycleInfo(Iterable<SkyKey> cycle) {
    return new CycleInfoWithDetails(ImmutableList.of(), cycle);
  }

  public static CycleInfo createCycleInfo(Iterable<SkyKey> pathToCycle, Iterable<SkyKey> cycle) {
    return new CycleInfoWithDetails(pathToCycle, cycle);
  }

  private static final CycleInfoNoDetails NO_DETAILS_INSTANCE = new CycleInfoNoDetails();

  public static CycleInfo cycleInfoNoDetails() {
    return NO_DETAILS_INSTANCE;
  }

  // Given a cycle and a value, if the value is part of the cycle, shift the cycle. Otherwise,
  // prepend the value to the head of pathToCycle.
  @Nullable
  private static CycleInfo normalizeCycle(final SkyKey value, CycleInfo cycle) {
    int index = cycle.getCycle().indexOf(value);
    if (index > -1) {
      if (!cycle.getPathToCycle().isEmpty()) {
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
      return new CycleInfoWithDetails(cycle.getCycle(), index);
    }
    return createCycleInfo(
        ImmutableList.<SkyKey>builderWithExpectedSize(cycle.getPathToCycle().size() + 1)
            .add(value)
            .addAll(cycle.getPathToCycle())
            .build(),
        cycle.getCycle());
  }

  /**
   * Normalize multiple cycles. This includes removing multiple paths to the same cycle, so that a
   * value does not depend on the same cycle multiple ways through the same child value. Note that a
   * value can still depend on the same cycle multiple ways, it's just that each way must be through
   * a different child value (a path with a different first element).
   *
   * <p>If any of the given cycles are without details (created using {@link #cycleInfoNoDetails()})
   * then a single {@link #cycleInfoNoDetails} will be returned.
   */
  static Iterable<CycleInfo> prepareCycles(final SkyKey value, Iterable<CycleInfo> cycles) {
    final Set<ImmutableList<SkyKey>> alreadyDoneCycles = new HashSet<>();
    ImmutableList.Builder<CycleInfo> result = ImmutableList.builder();
    for (CycleInfo cycle : cycles) {
      if (!cycle.hasCycleDetails()) {
        return ImmutableList.of(cycle);
      }
      CycleInfo normalized = normalizeCycle(value, cycle);
      if (normalized != null && alreadyDoneCycles.add(normalized.getCycle())) {
        result.add(normalized);
      }
    }
    return result.build();
  }

  private static class CycleInfoNoDetails extends CycleInfo {
    @Override
    public ImmutableList<SkyKey> getCycle() {
      throw new UnsupportedOperationException("unexpected access of cycle details");
    }

    @Override
    public ImmutableList<SkyKey> getPathToCycle() {
      throw new UnsupportedOperationException("unexpected access of cycle details");
    }

    @Override
    public SkyKey getTopKey() {
      throw new UnsupportedOperationException("unexpected access of cycle details");
    }

    @Override
    public boolean hasCycleDetails() {
      return false;
    }
  }

  private static class CycleInfoWithDetails extends CycleInfo {
    private final ImmutableList<SkyKey> cycle;
    private final ImmutableList<SkyKey> pathToCycle;

    private CycleInfoWithDetails(Iterable<SkyKey> pathToCycle, Iterable<SkyKey> cycle) {
      this.pathToCycle = ImmutableList.copyOf(pathToCycle);
      this.cycle = ImmutableList.copyOf(cycle);
      checkArgument(!this.cycle.isEmpty(), "Cycle cannot be empty: %s", this);
    }

    // If a cycle is already known, but we are processing a value in the middle of the cycle, we
    // need to shift the cycle so that the value is at the head.
    private CycleInfoWithDetails(Iterable<SkyKey> cycle, int cycleStart) {
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

    @Override
    public ImmutableList<SkyKey> getCycle() {
      return cycle;
    }

    @Override
    public ImmutableList<SkyKey> getPathToCycle() {
      return pathToCycle;
    }

    @Override
    public SkyKey getTopKey() {
      return pathToCycle.isEmpty() ? cycle.get(0) : pathToCycle.get(0);
    }

    @Override
    public boolean hasCycleDetails() {
      return true;
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
      if (!(that instanceof CycleInfoWithDetails thatCycle)) {
        return false;
      }

      return thatCycle.cycle.equals(this.cycle) && thatCycle.pathToCycle.equals(this.pathToCycle);
    }

    @Override
    public String toString() {
      return Iterables.toString(pathToCycle) + " -> " + Iterables.toString(cycle);
    }
  }
}
