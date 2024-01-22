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

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.ArrayDeque;
import java.util.HashSet;
import java.util.Queue;
import java.util.Set;
import java.util.function.Function;

/**
 * SkyframeFocuser is a minimizing optimizer (i.e. garbage collector) for the Skyframe graph, based
 * on a set of known inputs known as a working set, while ensuring correct incremental builds.
 */
public final class SkyframeFocuser {

  private SkyframeFocuser() {}

  /**
   * The set of SkyKeys kept after focusing. The actual change is done in place with the in-memory
   * graph.
   */
  public static class FocusResult {

    private final ImmutableSet<SkyKey> rdeps;
    private final ImmutableSet<SkyKey> deps;

    private FocusResult(ImmutableSet<SkyKey> rdeps, ImmutableSet<SkyKey> deps) {
      this.rdeps = rdeps;
      this.deps = deps;
    }

    /**
     * Returns the set of SkyKeys that are in the dependencies of all roots, and rdeps from the
     * leafs. May contain transitive dependencies, in cases where certain functions use them without
     * establishing a Skyframe dependency.
     */
    public ImmutableSet<SkyKey> getDeps() {
      return deps;
    }

    /** Returns the set of SkyKeys that are in the reverse dependencies of the leafs. */
    public ImmutableSet<SkyKey> getRdeps() {
      return rdeps;
    }
  }

  /**
   * Minimize the Skyframe graph by traverse it to prune nodes and edges that are not necessary for
   * the build correctness of a working set of files. The graph focusing algorithm pseudocode is as
   * follows.
   *
   * <ol>
   *   <li>Mark all the leafs and their transitive rdeps. For each marked node, also mark all their
   *       direct dependencies. An injectable function can also mark additional nodes reachable from
   *       the node itself.
   *   <li>For each marked node, remove all direct deps edges. Also remove all rdep edges unless
   *       they point to a rdep that should be kept. This creates the "flattened verification set".
   * </ol>
   *
   * @param graph the in-memory skyframe graph.
   * @param roots the SkyKeys of the roots to be kept, i.e. the top level keys.
   * @param leafs the SkyKeys of the leafs to be kept. This is the "working set".
   * @param eventHandler event handler to report stats during focusing
   * @param additionalDepsToKeep by default, all direct deps of rdeps are kept. this function is
   *     applied on all direct deps, in case there are optimizations elsewhere in the skyframe
   *     implementation that reads transitive nodes without specifying a dependency on them.
   * @return the set of kept SkyKeys in the in-memory graph, categorized by deps and rdeps.
   */
  public static FocusResult focus(
      InMemoryGraph graph,
      Set<SkyKey> roots,
      Set<SkyKey> leafs,
      EventHandler eventHandler,
      Function<SkyKey, Set<SkyKey>> additionalDepsToKeep) {
    Set<SkyKey> keptDeps = new HashSet<>();
    Set<SkyKey> keptRdeps = new HashSet<>();

    Queue<SkyKey> queue = new ArrayDeque<>();
    for (SkyKey leaf : leafs) {
      keptRdeps.add(leaf);
      queue.offer(leaf);
    }

    // Some roots are re-evaluated on every build. These roots may not be in the reverse TC
    // of leafs (working set), but may influence how the working set is evaluated (e.g. platform
    // mapping). If we remove them from the graph, those keys may be re-evaluated anyway (along with
    // their TC) on subsequent invocations, leading to wasted compute and RAM.
    // The exercise of ensuring which roots should be kept is left to the caller of
    // this function, but we ensure that all specified ones are kept here.
    keptDeps.addAll(roots);

    try (SilentCloseable c = Profiler.instance().profile("focus.mark")) {
      while (!queue.isEmpty()) {
        SkyKey key = queue.remove();
        InMemoryNodeEntry nodeEntry = graph.getIfPresent(key);
        if (nodeEntry == null) {
          // Throwing may be too strong if the working set is loosely defined, e.g. an entire
          // directory instead of a single file, and there are some files in the directory that
          // are not in the TC of the roots.
          //
          // TODO: b/312819241 - handle this gracefully without throwing.
          throw new IllegalStateException("nodeEntry not found for: " + key.getCanonicalName());
        }

        if (!nodeEntry.isDone()) {
          // TODO: b/312819241 - handle this gracefully without throwing.
          throw new IllegalStateException("nodeEntry not done: " + key.getCanonicalName());
        }

        for (SkyKey rdep : nodeEntry.getReverseDepsForDoneEntry()) {
          if (!keptRdeps.add(rdep)) {
            // Already processed.
            continue;
          }
          // Traverse up the graph.
          queue.offer(rdep);
        }

        for (SkyKey dep : nodeEntry.getDirectDeps()) {
          keptDeps.add(dep);

          // This is necessary to keep the action inputs encapsulated by a NestedSet. Otherwise,
          // those inputs will be missing.
          //
          // TODO: b/312819241 - move SkyframeFocuser from build.skyframe to build.lib.skyframe so
          // that we can do the check directly without using an injected Function.
          keptDeps.addAll(additionalDepsToKeep.apply(dep));
        }
      }
    }

    keptDeps.removeAll(keptRdeps);

    eventHandler.handle(
        Event.info("Nodes in reverse transitive closure from leafs: " + keptRdeps.size()));
    eventHandler.handle(
        Event.info("Nodes in direct deps of reverse transitive closure: " + keptDeps.size()));

    try (SilentCloseable c = Profiler.instance().profile("focus.sweep_nodes")) {
      Set<SkyKey> toKeep = Sets.union(keptDeps, keptRdeps);
      graph.parallelForEach(
          inMemoryNodeEntry -> {
            SkyKey key = inMemoryNodeEntry.getKey();
            if (!toKeep.contains(key)) {
              graph.remove(key);
            }
          });

      graph.shrinkNodeMap();
    }

    long rdepEdgesBefore = 0;
    long rdepEdgesAfter = 0;

    try (SilentCloseable c = Profiler.instance().profile("focus.sweep_edges")) {
      for (SkyKey key : keptDeps) {
        // TODO: b/312819241 - Consider transforming IncrementalInMemoryNodeEntry only used for
        // their
        // immutable states to an ImmutableDoneNodeEntry or NonIncrementalInMemoryNodeEntry for
        // further memory savings.
        IncrementalInMemoryNodeEntry nodeEntry =
            (IncrementalInMemoryNodeEntry) graph.getIfPresent(key);

        // No need to keep the direct deps edges of existing deps. For example:
        //
        //    B
        //  / |\
        // A C  \
        //    \ |
        //     D
        //
        // B is the root, and A is the only leaf. We can throw out the CD edge, even
        // though both C and D are still used by B. This is because no changes are expected to C
        // and D, so it's unnecessary to maintain the edges.
        nodeEntry.directDeps = GroupedDeps.EMPTY_COMPRESSED;

        // No need to keep the rdep edges of the deps if they do not point to an rdep
        // reachable (hence, dirty-able) by the working set.
        //
        // This accounts for nearly 5% of 9+GB retained heap on a large server build.
        rdepEdgesBefore += nodeEntry.getReverseDepsForDoneEntry().size();
        for (SkyKey rdep : nodeEntry.getReverseDepsForDoneEntry()) {
          if (!keptRdeps.contains(rdep)) {
            nodeEntry.removeReverseDep(rdep);
          }
        }

        // This consolidation is also done in getReverseDepsForDoneEntry(), but make it
        // explicit here (and it's idempotent, anyway).
        ReverseDepsUtility.consolidateData(nodeEntry);
        rdepEdgesAfter += nodeEntry.getReverseDepsForDoneEntry().size();
      }
    }

    eventHandler.handle(
        Event.info(String.format("Rdep edges: %s -> %s", rdepEdgesBefore, rdepEdgesAfter)));

    return new FocusResult(ImmutableSet.copyOf(keptRdeps), ImmutableSet.copyOf(keptDeps));
  }
}
