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

import static java.util.concurrent.TimeUnit.MINUTES;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.concurrent.AbstractQueueVisitor;
import com.google.devtools.build.lib.concurrent.ErrorClassifier;
import com.google.devtools.build.lib.events.Event;
import com.google.devtools.build.lib.events.EventHandler;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import java.util.Collection;
import java.util.Set;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/**
 * SkyframeFocuser is a minimizing optimizer (i.e. garbage collector) for the Skyframe graph, based
 * on a set of known inputs known as a working set, while ensuring correct incremental builds.
 *
 * <p>This is also a subclass of {@link AbstractQueueVisitor} to take advantage of highly
 * parallelizable operations over the Skyframe graph.
 */
public final class SkyframeFocuser extends AbstractQueueVisitor {

  // The in-memory Skyframe graph
  private final InMemoryGraph graph;

  // Event handler to report stats during focusing
  private final EventHandler eventHandler;

  private SkyframeFocuser(InMemoryGraph graph, EventHandler eventHandler) {
    super(
        /* parallelism= */ Runtime.getRuntime().availableProcessors(),
        /* keepAliveTime= */ 2,
        MINUTES,
        ExceptionHandlingMode.FAIL_FAST,
        /* poolName= */ "skyframe-focuser",
        ErrorClassifier.DEFAULT);
    this.graph = graph;
    this.eventHandler = eventHandler;
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
   * @param graph the in-memory graph to operate on
   * @param eventHandler handler to report events during focusing
   * @param roots the SkyKeys of the roots to be kept, i.e. the top level keys.
   * @param leafs the SkyKeys of the leafs to be kept. This is the "working set".
   * @param additionalDepsToKeep by default, all direct deps of rdeps are kept. this function is
   *     applied on all direct deps, in case there are optimizations elsewhere in the skyframe
   *     implementation that reads transitive nodes without specifying a dependency on them.
   * @return the set of kept SkyKeys in the in-memory graph, categorized by deps and rdeps.
   */
  public static FocusResult focus(
      InMemoryGraph graph,
      EventHandler eventHandler,
      Set<SkyKey> roots,
      Set<SkyKey> leafs,
      Function<SkyKey, Set<SkyKey>> additionalDepsToKeep)
      throws InterruptedException {
    SkyframeFocuser focuser = new SkyframeFocuser(graph, eventHandler);
    return focuser.run(roots, leafs, additionalDepsToKeep);
  }

  /**
   * The set of SkyKeys kept after focusing. The actual change is done in place with the in-memory
   * graph.
   */
  public static class FocusResult {

    private final ImmutableSet<SkyKey> roots;

    private final ImmutableSet<SkyKey> leafs;

    private final ImmutableSet<SkyKey> rdeps;

    private final ImmutableSet<SkyKey> deps;

    private FocusResult(
        ImmutableSet<SkyKey> roots,
        ImmutableSet<SkyKey> leafs,
        ImmutableSet<SkyKey> rdeps,
        ImmutableSet<SkyKey> deps) {
      this.roots = roots;
      this.leafs = leafs;
      this.rdeps = rdeps;
      this.deps = deps;
    }

    public ImmutableSet<SkyKey> getRoots() {
      return roots;
    }

    public ImmutableSet<SkyKey> getLeafs() {
      return leafs;
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
   * NodeVisitor is parallelizable graph visitor that's applied transitively from leafs to the
   * roots, while marking rdeps and all direct deps of those rdeps to be kept by SkyframeFocuser.
   */
  private class NodeVisitor implements Runnable {

    // The SkyKey that this NodeVisitor is responsible for.
    private final SkyKey key;

    // Threadsafe set of keys that depend on this key. May be modified by multiple NodeVisitors
    // concurrently.
    private final Set<SkyKey> keptRdeps;

    // Threadsafe set of keys that this key depends on. May be modified by multiple NodeVisitors
    // concurrently.
    private final Set<SkyKey> keptDeps;

    // See javadoc for the additionalDepsToKeep parameter of {@code SkyframeFocuser#focus}.
    private final Function<SkyKey, Set<SkyKey>> additionalDepsToKeep;

    protected NodeVisitor(
        SkyKey key,
        Set<SkyKey> keptRdeps,
        Set<SkyKey> keptDeps,
        Function<SkyKey, Set<SkyKey>> additionalDepsToKeep) {
      this.key = key;
      this.keptRdeps = keptRdeps;
      this.keptDeps = keptDeps;
      this.additionalDepsToKeep = additionalDepsToKeep;
    }

    @Override
    public void run() {
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
          // Memoization. Already processed.
          continue;
        }

        // Queue a traversal up the graph. This will not create duplicate NodeVisitors on the
        // same rdep due to the atomic keptRdeps.add check above.
        execute(new NodeVisitor(rdep, keptRdeps, keptDeps, additionalDepsToKeep));
      }

      for (SkyKey dep : nodeEntry.getDirectDeps()) {
        if (!keptDeps.add(dep)) {
          // Memoization. Already processed.
          continue;
        }

        // This is necessary to keep the action inputs encapsulated by a NestedSet. Otherwise,
        // those inputs will be missing.
        //
        // TODO: b/312819241 - move SkyframeFocuser from build.skyframe to build.lib.skyframe so
        // that we can do the check directly without using an injected Function.
        execute(() -> keptDeps.addAll(additionalDepsToKeep.apply(dep)));
      }
    }
  }

  /** Entry point of the Skyframe garbage collection algorithm. */
  private FocusResult run(
      Set<SkyKey> roots, Set<SkyKey> leafs, Function<SkyKey, Set<SkyKey>> additionalDepsToKeep)
      throws InterruptedException {

    Set<SkyKey> keptDeps = Sets.newConcurrentHashSet();
    Set<SkyKey> keptRdeps = Sets.newConcurrentHashSet();

    // All leafs are automatically considered as rdeps.
    keptRdeps.addAll(leafs);

    // All roots are automatically considered as deps.
    //
    // Some roots are re-evaluated on every build. These roots may not be in the reverse TC
    // of leafs (working set), but may influence how the working set is evaluated (e.g. platform
    // mapping). If we remove them from the graph, those keys may be re-evaluated anyway (along with
    // their TC) on subsequent invocations, leading to wasted compute and RAM.
    // The exercise of ensuring which roots should be kept is left to the caller of
    // this function, but we ensure that all specified ones are kept here.
    keptDeps.addAll(roots);

    try (SilentCloseable c = Profiler.instance().profile("focus.mark")) {
      // Start traversal from leafs.
      for (SkyKey leaf : leafs) {
        execute(new NodeVisitor(leaf, keptRdeps, keptDeps, additionalDepsToKeep));
      }
      awaitQuiescenceWithoutShutdown(true);
    }

    // Keep the rdeps transitive closure from leafs distinct from the deps.
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

    AtomicLong rdepEdgesBefore = new AtomicLong();
    AtomicLong rdepEdgesAfter = new AtomicLong();

    try (SilentCloseable c = Profiler.instance().profile("focus.sweep_edges")) {
      for (SkyKey key : keptDeps) {
        execute(
            () -> {
              // TODO: b/312819241 - Consider transforming IncrementalInMemoryNodeEntry only used
              // for their immutable states to an ImmutableDoneNodeEntry or
              // NonIncrementalInMemoryNodeEntry
              // for further memory savings.
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
              // though both C and D are still used by B. This is because no changes are expected to
              // C and D, so it's unnecessary to maintain the edges.
              nodeEntry.directDeps = GroupedDeps.EMPTY_COMPRESSED;

              // No need to keep the rdep edges of the deps if they do not point to an rdep
              // reachable (hence, dirty-able) by the working set.
              //
              // This accounts for nearly 5% of 9+GB retained heap on a large server build.
              Collection<SkyKey> existingRdeps = nodeEntry.getReverseDepsForDoneEntry();
              rdepEdgesBefore.getAndAdd(existingRdeps.size());
              int rdepEdgesKept = 0;
              for (SkyKey rdep : existingRdeps) {
                if (keptRdeps.contains(rdep)) {
                  rdepEdgesKept++;
                } else {
                  nodeEntry.removeReverseDep(rdep);
                }
              }
              rdepEdgesAfter.getAndAdd(rdepEdgesKept);

              ReverseDepsUtility.consolidateData(nodeEntry);
            });
      }

      awaitQuiescence(true); // and shut down the ExecutorService.
    }

    eventHandler.handle(
        Event.info(
            String.format("Rdep edges: %s -> %s", rdepEdgesBefore.get(), rdepEdgesAfter.get())));

    return new FocusResult(
        ImmutableSet.copyOf(roots),
        ImmutableSet.copyOf(leafs),
        ImmutableSet.copyOf(keptRdeps),
        ImmutableSet.copyOf(keptDeps));
  }
}
