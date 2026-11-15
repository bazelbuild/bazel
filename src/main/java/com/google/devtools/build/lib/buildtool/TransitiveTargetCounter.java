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
package com.google.devtools.build.lib.buildtool;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.RuleConfiguredObjectValue;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.NamedForkJoinPool;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ActionLookupValuesTraversal;
import com.google.devtools.build.lib.skyframe.AspectKeyCreator.AspectBaseKey;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Arrays;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.ForkJoinTask;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.stream.IntStream;
import javax.annotation.Nullable;

/**
 * Counts the number of analysis nodes ({@link RuleConfiguredObjectValue}s) and actions in the
 * transitive closure of each root target.
 *
 * <p>Key insight: summing counts bottom-up would double-count shared diamond deps, while a separate
 * traversal per root would revisit shared subgraphs multiple times. Instead, this algorithm flips
 * the question around (from "which nodes does root r reach" to "which roots reach node n"?). This
 * is propagated top-down via a bitmask (64 roots per word), making the operation cheap bitwise ORs.
 *
 * <p>The algorithm has two passes, both of which are fork/join traversals with one small task per
 * node:
 *
 * <ol>
 *   <li><b>Discovery</b> (runs once) walks the Skyframe graph from {@code roots} and builds a
 *       compact DAG of the relevant {@link ActionLookupKey}s indexed by dense IDs, with each node's
 *       action count, integer-indexed dependencies and in-degree.
 *   <li><b>Propagation</b> (runs in batches of roots) computes, for every node, the set of roots it
 *       is reachable from as a bitmask. A node is processed once all of its parents have OR-ed
 *       their masks into it (Kahn's algorithm), which yields a topological order on the fly with no
 *       separate sorting pass. Processing a node adds its counts to every root in its mask.
 * </ol>
 */
final class TransitiveTargetCounter {

  /** Transitive analysis node and action counts for a single root target. */
  record Counts(long targetCount, long actionCount) {}

  /** Atomic element access on {@link Propagation#state} ({@code long[]}). */
  private static final VarHandle LONG_ARRAY = MethodHandles.arrayElementVarHandle(long[].class);

  /**
   * Atomic operations on {@link Node#indegree} (a plain {@code int} field), avoiding a separate
   * {@link AtomicInteger} allocation per {@link Node}.
   */
  private static final VarHandle INDEGREE;

  static {
    try {
      INDEGREE = MethodHandles.lookup().findVarHandle(Node.class, "indegree", int.class);
    } catch (ReflectiveOperationException e) {
      throw new ExceptionInInitializerError(e);
    }
  }

  /** Upper bound on the size of the per-pass bitmask array, to bound memory on huge inputs. */
  private static final long MAX_MASK_BYTES = 1L << 27;

  /** Parallelism threshold for {@link ConcurrentHashMap#forEachValue}. */
  private static final long FOR_EACH_PARALLELISM_THRESHOLD = 1024;

  private static final int[] NO_DEPS = new int[0];

  private TransitiveTargetCounter() {}

  /** A node of the compact subgraph. */
  private static final class Node {
    final int id;

    /** Number of actions, or -1 if this node is not a {@link RuleConfiguredObjectValue}. */
    int actions = -1;

    /** IDs of the relevant direct dependencies. */
    int[] deps = NO_DEPS;

    /** Number of incoming edges from other nodes; updated via {@link #INDEGREE}. */
    int indegree;

    Node(int id) {
      this.id = id;
    }
  }

  /**
   * Returns {@code obj} as an {@link ActionLookupKey} if it can be or transitively reach a {@link
   * RuleConfiguredObjectValue}, or {@code null} otherwise.
   *
   * <p>Non-configurable {@link ConfiguredTargetKey}s ({@code configurationKey == null}, i.e. {@code
   * InputFile}, {@code PackageGroup}, {@code EnvironmentGroup}) always produce {@code
   * NonRuleConfiguredTargetValue} and only depend on visibility {@code PackageGroup}s, so they can
   * never be or reach a {@link RuleConfiguredObjectValue}.
   */
  @Nullable
  private static ActionLookupKey asRelevantKey(@Nullable Object obj) {
    if (obj instanceof ConfiguredTargetKey ctk) {
      return ctk.getConfigurationKey() != null ? ctk : null;
    }
    if (obj instanceof AspectBaseKey aspectKey) {
      return aspectKey;
    }
    return null;
  }

  /** Shared state and traversal logic of the discovery pass. */
  private static final class Discovery {
    private final InMemoryGraph graph;
    final ConcurrentHashMap<ActionLookupKey, Node> nodes;
    final AtomicInteger nextId = new AtomicInteger();

    Discovery(InMemoryGraph graph) {
      this.graph = graph;
      this.nodes = new ConcurrentHashMap<>(Math.max(1024, graph.valuesSize() / 2));
    }

    /**
     * Returns the node of {@code key}, creating it and forking a {@link VisitTask} under {@code
     * parent} if this is the first time the key is seen.
     */
    Node nodeFor(ActionLookupKey key, CountedCompleter<?> parent) {
      return nodes.computeIfAbsent(
          key,
          k -> {
            Node fresh = new Node(nextId.getAndIncrement());
            parent.addToPendingCount(1);
            new VisitTask(parent, this, k, fresh).fork();
            return fresh;
          });
    }

    /** Visits one node: records its actions and dependencies and forks tasks for new deps. */
    void visit(CountedCompleter<?> parent, ActionLookupKey key, Node node) {
      InMemoryNodeEntry entry = graph.getIfPresent(key);
      if (entry != null && entry.isDone()) {
        SkyValue value = entry.getValue();
        if (value instanceof RuleConfiguredObjectValue ruleConfiguredObject
            && !ActionLookupValuesTraversal.isDelegatedConfiguredTargetValue(key, value)) {
          node.actions = ruleConfiguredObject.getActions().size();
        }
        // `value` is null when a done node completed with an error under --keep_going
        // (entry.getValue() unwraps ErrorInfoValue to null), including cycle nodes whose retained
        // cycle edges would violate the DAG invariant of Kahn's algorithm. `keepsEdges()` is false
        // under --notrack_incremental_state (NonIncrementalInMemoryNodeEntry), where direct deps
        // are discarded on completion and getCompressedDirectDepsForDoneEntry() throws.
        if (value != null && entry.keepsEdges()) {
          node.deps = visitCompressedDeps(parent, entry.getCompressedDirectDepsForDoneEntry());
        }
      }
    }

    /**
     * Visits the deps in a compressed {@code GroupedDeps}: either an {@code Object[]} of {@link
     * SkyKey}s and {@code Integer} group markers, a single bare {@link SkyKey}, or an opaque empty
     * marker.
     */
    private int[] visitCompressedDeps(CountedCompleter<?> parent, Object compressed) {
      if (!(compressed instanceof Object[] depArray)) {
        ActionLookupKey depKey = asRelevantKey(compressed);
        if (depKey == null) {
          return NO_DEPS;
        }
        Node dep = nodeFor(depKey, parent);
        INDEGREE.getAndAdd(dep, 1);
        return new int[] {dep.id};
      }
      // Look all dependencies up and read their IDs first, and only then bump their in-degrees:
      // reading dep.id faults the Node's cache line in alongside the hash-table lookup while the
      // out-of-order engine can overlap misses across iterations, whereas the atomic increments in
      // the second loop are full fences that would otherwise serialize them.
      int maxDeps = depArray.length;
      Node[] depNodes = new Node[maxDeps];
      int[] deps = new int[maxDeps];
      int n = 0;
      for (Object elem : depArray) {
        ActionLookupKey depKey = asRelevantKey(elem);
        if (depKey == null) {
          continue;
        }
        Node dep = nodeFor(depKey, parent);
        depNodes[n] = dep;
        deps[n] = dep.id;
        n++;
      }
      for (int i = 0; i < n; i++) {
        INDEGREE.getAndAdd(depNodes[i], 1);
      }
      return n == 0 ? NO_DEPS : (n == maxDeps ? deps : Arrays.copyOf(deps, n));
    }
  }

  /** Fork/join task that visits one node during {@link Discovery}. */
  private static final class VisitTask extends CountedCompleter<Void> {
    private final Discovery discovery;
    private final ActionLookupKey key;
    private final Node node;

    VisitTask(CountedCompleter<?> parent, Discovery discovery, ActionLookupKey key, Node node) {
      super(parent);
      this.discovery = discovery;
      this.key = key;
      this.node = node;
    }

    @Override
    public void compute() {
      discovery.visit(this, key, node);
      tryComplete();
    }
  }

  /**
   * Shared state and traversal logic of one propagation pass over the root range {@code
   * [iFirstRoot, iLastRoot)}.
   */
  private static final class Propagation {
    private final Node[] byId;
    private final int iFirstRoot;
    private final int iLastRoot;
    private final int words;

    /**
     * Number of {@code long} slots per node in {@link #state}: {@link #words} bitmask words plus 1
     * slot for the remaining in-degree.
     */
    private final int stride;

    /** Per node: {@code words} mask words followed by the remaining in-degree. */
    private final long[] state;

    /**
     * Thread-local {@code [targets..., actions...]} accumulators, one pair of slots per root of
     * this pass (index {@code i - iFirstRoot}).
     */
    private final ThreadLocal<long[]> accumulator;

    final ConcurrentLinkedQueue<long[]> accumulators = new ConcurrentLinkedQueue<>();

    Propagation(Node[] byId, int iFirstRoot, int iLastRoot) {
      this.byId = byId;
      this.iFirstRoot = iFirstRoot;
      this.iLastRoot = iLastRoot;
      this.words = (iLastRoot - iFirstRoot + 63) / 64;
      // + 1 for the remaining in-degree stored after each node's bitmask words.
      this.stride = words + 1;
      this.state = new long[byId.length * stride];
      this.accumulator =
          ThreadLocal.withInitial(
              () -> {
                long[] acc = new long[2 * (iLastRoot - iFirstRoot)];
                accumulators.add(acc);
                return acc;
              });
    }

    CountedCompleter<Void> createTask(Node[] rootNodes) {
      return new CountedCompleter<Void>() {
        @Override
        public void compute() {
          start(this, rootNodes);
          tryComplete();
        }
      };
    }

    /** Initializes {@link #state} and forks {@link PropagateTask}s for all in-degree-0 roots. */
    private void start(CountedCompleter<?> parent, Node[] rootNodes) {
      // Runs on a pool worker, so the parallel stream is executed by this pool too.
      IntStream.range(0, byId.length)
          .parallel()
          .forEach(id -> state[id * stride + words] = byId[id].indegree);
      for (int i = iFirstRoot; i < iLastRoot; i++) {
        int bit = i - iFirstRoot;
        state[rootNodes[i].id * stride + bit / 64] |= 1L << (bit % 64);
      }
      // Only roots can have no parents; start every in-degree-0 root across all roots (not just
      // [iFirstRoot, iLastRoot)) so Kahn's algorithm reaches roots in [iFirstRoot, iLastRoot)
      // whose parents are roots from other passes. Because `roots` is a Set, rootNodes are
      // distinct and each is started once.
      for (Node root : rootNodes) {
        if (root.indegree == 0) {
          parent.addToPendingCount(1);
          new PropagateTask(parent, this, root).fork();
        }
      }
    }

    /** Processes one node whose parents have all been processed and forks its ready children. */
    void propagate(CountedCompleter<?> parent, Node node) {
      int base = node.id * stride;

      if (node.actions >= 0) {
        long[] acc = accumulator.get();
        int passRoots = acc.length / 2;
        for (int w = 0; w < words; w++) {
          long mask = state[base + w];
          int rootBase = w * 64;
          // Iterate over the set bits in `mask`; where a bit is set, update the accumulator
          // accordingly.
          while (mask != 0) {
            int root = rootBase + Long.numberOfTrailingZeros(mask);
            acc[root]++;
            acc[passRoots + root] += node.actions;
            // Unset the lowest set bit.
            mask &= mask - 1;
          }
        }
      }

      // Iterate over every dep: OR the mask of this node into the mask of the dep and decrement the
      // remaining in-degree of the dep by one. If the dep has no more parents remaining, enqueue
      // it.
      for (int dep : node.deps) {
        int depBase = dep * stride;
        for (int w = 0; w < words; w++) {
          long mask = state[base + w];
          // Skip the (fenced) atomic OR if the child already has all of our bits, which is common
          // when parents share roots. Bits are only ever set, so a stale read is harmless.
          if (mask != 0) {
            long current = state[depBase + w];
            if ((current | mask) != current) {
              LONG_ARRAY.getAndBitwiseOr(state, depBase + w, mask);
            }
          }
        }
        if ((long) LONG_ARRAY.getAndAdd(state, depBase + words, -1L) == 1L) {
          parent.addToPendingCount(1);
          new PropagateTask(parent, this, byId[dep]).fork();
        }
      }
    }
  }

  /** Fork/join task that processes one ready node during {@link Propagation}. */
  private static final class PropagateTask extends CountedCompleter<Void> {
    private final Propagation propagation;
    private final Node node;

    PropagateTask(CountedCompleter<?> parent, Propagation propagation, Node node) {
      super(parent);
      this.propagation = propagation;
      this.node = node;
    }

    @Override
    public void compute() {
      propagation.propagate(this, node);
      tryComplete();
    }
  }

  /** Runs {@code task} in {@code pool}, propagating failures and interrupts. */
  private static void run(ForkJoinPool pool, ForkJoinTask<?> task) throws InterruptedException {
    try {
      pool.submit(task).get();
    } catch (ExecutionException e) {
      Throwables.throwIfUnchecked(e.getCause());
      throw new IllegalStateException(e.getCause());
    }
  }

  /**
   * Runs the discovery pass from {@code rootKeys}, filling {@code rootNodes} with their nodes, and
   * returns all discovered nodes indexed by ID.
   */
  private static Node[] discover(
      InMemoryGraph graph, ConfiguredTargetKey[] rootKeys, Node[] rootNodes, ForkJoinPool pool)
      throws InterruptedException {
    Discovery discovery = new Discovery(graph);
    CountedCompleter<Void> discoveryRoot =
        new CountedCompleter<Void>() {
          @Override
          public void compute() {
            for (int i = 0; i < rootKeys.length; i++) {
              rootNodes[i] = discovery.nodeFor(rootKeys[i], this);
            }
            tryComplete();
          }
        };
    run(pool, discoveryRoot);

    // Populate `byId` in a separate pass once `nextId` gives the exact reachable node count:
    // during discovery that count is unknown (it can be much smaller than `graph.valuesSize()`,
    // or larger if roots or deps are absent from `graph`), so building `byId` on the fly would
    // require concurrent resizing or chunked arrays. Wrapping `forEachValue` in
    // `ForkJoinTask.adapt` runs it in `pool` so its subtasks fork onto `pool` instead of
    // `ForkJoinPool.commonPool()`.
    Node[] byId = new Node[discovery.nextId.get()];
    run(
        pool,
        ForkJoinTask.adapt(
            () ->
                discovery.nodes.forEachValue(
                    FOR_EACH_PARALLELISM_THRESHOLD, node -> byId[node.id] = node)));
    return byId;
  }

  static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots) throws InterruptedException {
    return count(graph, roots, MAX_MASK_BYTES);
  }

  @VisibleForTesting
  static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots, long maxMaskBytes)
      throws InterruptedException {
    if (roots.isEmpty()) {
      return ImmutableMap.of();
    }

    ForkJoinPool pool =
        NamedForkJoinPool.newNamedPool(
            "transitive-target-counter", Runtime.getRuntime().availableProcessors());
    boolean completedNormally = false;
    try {
      ImmutableMap<ConfiguredTargetKey, Counts> result = count(graph, roots, pool, maxMaskBytes);
      completedNormally = true;
      return result;
    } finally {
      if (completedNormally
          ? ExecutorUtil.interruptibleShutdown(pool)
          : ExecutorUtil.uninterruptibleShutdownNow(pool)) {
        Thread.currentThread().interrupt();
      }
    }
  }

  private static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots, ForkJoinPool pool, long maxMaskBytes)
      throws InterruptedException {
    ConfiguredTargetKey[] rootKeys = roots.toArray(new ConfiguredTargetKey[0]);
    int numRoots = rootKeys.length;
    long[] totals = new long[2 * numRoots];

    // 1. Discovery, builds up a compact DAG indexed by dense integer IDs. Done in a separate method
    // so that the key -> Node map is unreachable (and collectable) during propagation.
    Node[] rootNodes = new Node[numRoots];
    Node[] byId;
    try (SilentCloseable c = Profiler.instance().profile("TransitiveTargetCounter.discovery")) {
      byId = discover(graph, rootKeys, rootNodes, pool);
    }

    // 2. Propagation, in passes over ranges of roots if the bitmasks would be too large.
    //
    //   roots:  [0 ................. iFirstRoot .... iLastRoot ................. numRoots)
    //                                |<-- <= 64*words roots -->|
    //                                  bit b = i - iFirstRoot -> word b/64, bit b%64
    //
    //   state:  |<- node 0 (stride = words + 1) ->|<- node 1 (stride = words + 1) ->| ...
    //           +---------+-----+-----------+-----+---------+-----+-----------+-----+
    //           | mask[0] | ... | mask[w-1] |indeg| mask[0] | ... | mask[w-1] |indeg| ...
    //           +---------+-----+-----------+-----+---------+-----+-----------+-----+
    //
    //   totals: |<--- targetCount[0..numRoots) --->|<--- actionCount[0..numRoots) --->|
    try (SilentCloseable c = Profiler.instance().profile("TransitiveTargetCounter.propagation")) {
      int nodeCount = byId.length;
      int maxWordsPerPass =
          (int) Math.max(1, Math.min(1 << 20, maxMaskBytes / (8L * Math.max(1, nodeCount))));
      for (int iFirstRoot = 0; iFirstRoot < numRoots; iFirstRoot += 64 * maxWordsPerPass) {
        int iLastRoot = Math.min(numRoots, iFirstRoot + 64 * maxWordsPerPass);
        int passRoots = iLastRoot - iFirstRoot;
        Propagation propagation = new Propagation(byId, iFirstRoot, iLastRoot);
        run(pool, propagation.createTask(rootNodes));
        for (long[] acc : propagation.accumulators) {
          for (int i = 0; i < passRoots; i++) {
            totals[iFirstRoot + i] += acc[i];
            totals[numRoots + iFirstRoot + i] += acc[passRoots + i];
          }
        }
      }
    }

    ImmutableMap.Builder<ConfiguredTargetKey, Counts> result =
        ImmutableMap.builderWithExpectedSize(numRoots);
    for (int i = 0; i < numRoots; i++) {
      result.put(rootKeys[i], new Counts(totals[i], totals[numRoots + i]));
    }
    return result.buildOrThrow();
  }
}
