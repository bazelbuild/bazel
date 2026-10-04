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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.flogger.GoogleLogger;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Prototype of speculative dependencies: the dependencies that nodes requested speculatively during
 * an evaluation and those that the cycle detector cut to break a cycle.
 *
 * <p>An instance is shared by all rounds of an evaluation: a cut ends the round, the nodes that were
 * in flight are deleted and evaluated again in a new round, in which the node that requested the
 * cut dependency sees it in {@link SkyFunction.Environment#getCutSpeculativeDeps}.
 */
final class SpeculativeDeps {
  private static final GoogleLogger logger = GoogleLogger.forEnclosingClass();

  private final ConcurrentHashMap<SkyKey, Set<SkyKey>> requested = new ConcurrentHashMap<>();
  private final ConcurrentHashMap<SkyKey, Set<SkyKey>> cut = new ConcurrentHashMap<>();
  private final AtomicBoolean cutInRound = new AtomicBoolean();

  void noteRequested(SkyKey parent, Iterable<SkyKey> deps) {
    Set<SkyKey> set = requested.computeIfAbsent(parent, k -> ConcurrentHashMap.newKeySet());
    for (SkyKey dep : deps) {
      set.add(dep);
    }
  }

  ImmutableSet<SkyKey> getCut(SkyKey parent) {
    Set<SkyKey> set = cut.get(parent);
    return set == null ? ImmutableSet.of() : ImmutableSet.copyOf(set);
  }

  /**
   * Cuts the first edge of the given cycle whose parent requested the child speculatively.
   *
   * @return whether an edge was cut
   */
  boolean cutEdgeOf(ImmutableList<SkyKey> cycle) {
    for (int i = 0; i < cycle.size(); i++) {
      SkyKey parent = cycle.get(i);
      SkyKey child = cycle.get((i + 1) % cycle.size());
      Set<SkyKey> set = requested.get(parent);
      if (set == null || !set.contains(child)) {
        continue;
      }
      logger.atInfo().log(
          "Cutting speculative dependency of %s on %s in cycle %s", parent, child, cycle);
      cut.computeIfAbsent(parent, k -> ConcurrentHashMap.newKeySet()).add(child);
      cutInRound.set(true);
      return true;
    }
    return false;
  }

  /** Returns whether a dependency was cut in the current round of the evaluation. */
  boolean hasCut() {
    return cutInRound.get();
  }

  /** Returns whether a dependency was cut in the round that ends with this call. */
  boolean consumeCut() {
    return cutInRound.getAndSet(false);
  }
}
