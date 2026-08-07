// Copyright 2026 The Bazel Authors. All rights reserved.
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
import com.google.devtools.build.lib.concurrent.ForkJoinQuiescingExecutor;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Finds the dirty nodes that change pruning would verify clean if it ran, which it doesn't purely
 * because these nodes weren't in scope for the current evaluation.
 *
 * <p>Operates in two phases: a parallel scan over all candidates that builds a DAG of dirty but
 * unchanged deletion candidates and their dirty rdeps, followed by a traversal of the DAG that
 * keeps all nodes whose dirty deps are also kept. The scan performs all graph accesses and its cost
 * scales with the total number of candidates, so it runs on multiple threads; the traversal only
 * visits the nodes that end up being kept.
 */
final class ChangePrunableNodesFinder {
  private static final int SCAN_THREAD_COUNT = Runtime.getRuntime().availableProcessors();

  private final InMemoryGraph graph;
  private final ImmutableSet<SkyKey> candidates;

  // The DAG built by the scan phase: for each candidate with at least one dirty dep, the number
  // of its dirty deps not yet known to be kept, together with the reverse edges. Candidates
  // without dirty deps start out ready. Each entry in remainingDirtyDeps is written by a single
  // scan thread, whereas dirtyRdeps values may receive edges from multiple threads.
  private final ConcurrentHashMap<SkyKey, AtomicInteger> remainingDirtyDeps =
      new ConcurrentHashMap<>();
  private final ConcurrentHashMap<SkyKey, ArrayList<SkyKey>> dirtyRdeps = new ConcurrentHashMap<>();
  private final ConcurrentLinkedQueue<SkyKey> ready = new ConcurrentLinkedQueue<>();

  ChangePrunableNodesFinder(InMemoryGraph graph, ImmutableSet<SkyKey> candidates) {
    this.graph = graph;
    this.candidates = candidates;
  }

  ImmutableSet<SkyKey> find() {
    ImmutableList<SkyKey> candidateList = candidates.asList();
    if (candidateList.isEmpty()) {
      return ImmutableSet.of();
    }

    var executor =
        ForkJoinQuiescingExecutor.newBuilder()
            .withOwnershipOf(
                NamedForkJoinPool.newNamedPool("find-change-prunable-nodes", SCAN_THREAD_COUNT))
            .build();
    long listSize = candidateList.size();
    long numJobs = Math.min(SCAN_THREAD_COUNT, listSize);
    for (long i = 0; i < numJobs; i++) {
      int startIndex = (int) ((i * listSize) / numJobs);
      int endIndex = (int) (((i + 1) * listSize) / numJobs);
      List<SkyKey> chunk = candidateList.subList(startIndex, endIndex);
      executor.execute(() -> scanCandidates(chunk));
    }
    try {
      executor.awaitQuiescence(/* interruptWorkers= */ true);
    } catch (InterruptedException e) {
      // Keeping no nodes is always safe, it merely GCs more aggressively.
      Thread.currentThread().interrupt();
      return ImmutableSet.of();
    }

    // Now visit the DAG, keeping all nodes whose dirty deps are also kept.
    var toKeep = ImmutableSet.<SkyKey>builder();
    SkyKey readyKey;
    while ((readyKey = ready.poll()) != null) {
      toKeep.add(readyKey);
      var rdeps = dirtyRdeps.get(readyKey);
      if (rdeps == null) {
        continue;
      }
      for (var rdep : rdeps) {
        if (remainingDirtyDeps.get(rdep).decrementAndGet() == 0) {
          // The last dirty dep of rdep is now known to be kept.
          ready.add(rdep);
        }
      }
    }
    return toKeep.build();
  }

  private void scanCandidates(List<SkyKey> chunk) {
    skipCandidate:
    for (var skyKey : chunk) {
      if (Thread.currentThread().isInterrupted()) {
        // A partial scan only ever results in fewer nodes being kept, which is always safe.
        return;
      }
      if (!(graph.getIfPresent(skyKey) instanceof IncrementalInMemoryNodeEntry entry)) {
        continue;
      }
      Iterable<SkyKey> lastBuildDeps = entry.lastBuildDepsIfChangePrunable();
      if (lastBuildDeps == null) {
        continue;
      }
      Version lastEvaluated = entry.lastEvaluatedVersion();
      var dirtyDeps = new ArrayList<SkyKey>();
      for (var dep : lastBuildDeps) {
        NodeEntry depEntry = graph.getIfPresent(dep);
        // A dep must be present and unchanged since this node was last evaluated to be
        // potentially change-prunable. Undone deps that aren't unenqueued dirty deps will never
        // be considered below and thus make an entry not change-prunable.
        if (depEntry == null || !depEntry.getVersion().atMost(lastEvaluated)) {
          continue skipCandidate;
        }
        if (depEntry.isDone()) {
          continue;
        }
        if (!candidates.contains(dep)) {
          continue skipCandidate;
        }
        dirtyDeps.add(dep);
      }
      if (dirtyDeps.isEmpty()) {
        ready.add(skyKey);
      } else {
        remainingDirtyDeps.put(skyKey, new AtomicInteger(dirtyDeps.size()));
        for (var dirtyDep : dirtyDeps) {
          dirtyRdeps.compute(
              dirtyDep,
              (unusedKey, rdeps) -> {
                if (rdeps == null) {
                  rdeps = new ArrayList<>();
                }
                rdeps.add(skyKey);
                return rdeps;
              });
        }
      }
    }
  }
}
