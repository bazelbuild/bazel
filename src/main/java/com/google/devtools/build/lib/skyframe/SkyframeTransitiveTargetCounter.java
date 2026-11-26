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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableMap;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.RuleConfiguredObjectValue;
import com.google.devtools.build.lib.collect.ObjectToIntMap;
import com.google.devtools.build.lib.concurrent.ForkJoinUtil;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.TransitiveClosureCounter.Counts;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.Set;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import javax.annotation.Nullable;

/**
 * Counts the number of analysis nodes ({@link RuleConfiguredObjectValue}s) and actions in the
 * transitive closure of each root target from the Skyframe graph.
 *
 * <p>The discovery pass, a fork/join traversal with one small task per node, walks the Skyframe
 * graph from {@code roots} and assigns dense IDs to the relevant {@link ActionLookupKey}s, with
 * each node's action count, in-degree and Skyframe node entry. {@link TransitiveClosureCounter}
 * then counts the closures, reading the dependencies of each node from its Skyframe node entry
 * rather than from a copy, to save memory.
 *
 * <p>Only works if Skyframe keeps the edges of the graph, i.e. if it tracks incremental state.
 */
public final class SkyframeTransitiveTargetCounter {

  /** Atomic element access on the {@code int[]} arrays of {@link Chunk}s. */
  private static final VarHandle INT_ARRAY = MethodHandles.arrayElementVarHandle(int[].class);

  private SkyframeTransitiveTargetCounter() {}

  private static final int CHUNK_BITS = 14;
  private static final int CHUNK_SIZE = 1 << CHUNK_BITS;
  private static final int CHUNK_MASK = CHUNK_SIZE - 1;

  /** The data of {@link #CHUNK_SIZE} consecutive discovered nodes. */
  private static final class Chunk {
    /** Number of actions + 1 per node; 0 if the node is not a {@link RuleConfiguredObjectValue}. */
    final int[] actionsPlusOne = new int[CHUNK_SIZE];

    /** Number of incoming edges from other nodes per node; updated via {@link #INT_ARRAY}. */
    final int[] indegrees = new int[CHUNK_SIZE];

    /** The Skyframe node entry whose direct deps are followed, or null if there are none. */
    final InMemoryNodeEntry[] entries = new InMemoryNodeEntry[CHUNK_SIZE];
  }

  /**
   * The per-node data of the discovered nodes, by ID, in {@link Chunk}s that are created on demand.
   *
   * <p>Chunks are needed because the number of nodes is only known once discovery is done, while
   * discovery threads write the data of the nodes they visit concurrently. A single array would
   * have to be reallocated and copied when it grows, which would mean either locking every access
   * or losing writes that race with the copy. Chunks are created lock-free and never move.
   */
  private static final class Nodes {
    // IDs are non-negative ints, so there are at most 2^(31 - CHUNK_BITS) chunks.
    final AtomicReferenceArray<Chunk> chunks = new AtomicReferenceArray<>(1 << (31 - CHUNK_BITS));

    /** Returns the chunk of node {@code id}, creating it if needed. */
    Chunk chunk(int id) {
      int index = id >>> CHUNK_BITS;
      Chunk chunk = chunks.get(index);
      if (chunk == null) {
        Chunk fresh = new Chunk();
        chunk = chunks.compareAndExchange(index, null, fresh);
        if (chunk == null) {
          chunk = fresh;
        }
      }
      return chunk;
    }

    void incrementIndegree(int id) {
      INT_ARRAY.getAndAdd(chunk(id).indegrees, id & CHUNK_MASK, 1);
    }
  }

  /** Returns {@code obj} as an {@link ActionLookupKey} if it is relevant, or {@code null}. */
  @Nullable
  private static ActionLookupKey asRelevantKey(@Nullable Object obj) {
    return TransitiveClosureCounter.isRelevant(obj) ? (ActionLookupKey) obj : null;
  }

  /** Shared state and traversal logic of the discovery pass. */
  private static final class Discovery {
    private final InMemoryGraph graph;
    final ObjectToIntMap<ActionLookupKey> ids = new ObjectToIntMap<>();
    final Nodes nodes = new Nodes();
    final AtomicInteger nextId = new AtomicInteger();

    Discovery(InMemoryGraph graph) {
      this.graph = graph;
    }

    /**
     * Returns the ID of {@code key}. If this is the first time the key is seen, assigns it a new ID
     * and forks a {@link VisitTask} under {@code parent} that visits the node and in turn does the
     * same for its direct deps.
     */
    int generateIdAndEnqueueDirectDeps(ActionLookupKey key, CountedCompleter<?> parent) {
      int id = ids.getOrPut(key, nextId::getAndIncrement);
      if (id >= 0) {
        // Seen before, so it is already visited or about to be.
        return id;
      }
      // getOrPut() returns the bitwise complement of the ID if it added the key, i.e. if this is
      // the first time the key is seen; only then is the node visited.
      id = ~id;
      parent.addToPendingCount(1);
      new VisitTask(parent, this, key, id).fork();
      return id;
    }

    /** Visits one node: records its actions and Skyframe entry and forks tasks for new deps. */
    void visit(CountedCompleter<?> parent, ActionLookupKey key, int id) {
      InMemoryNodeEntry entry = graph.getIfPresent(key);
      if (entry != null && entry.isDone()) {
        Chunk chunk = nodes.chunk(id);
        SkyValue value = entry.getValue();
        if (value instanceof RuleConfiguredObjectValue ruleConfiguredObject
            && !ActionLookupValuesTraversal.isDelegatedConfiguredTargetValue(key, value)) {
          chunk.actionsPlusOne[id & CHUNK_MASK] = ruleConfiguredObject.getActions().size() + 1;
        }
        // `value` is null when a done node completed with an error under --keep_going
        // (entry.getValue() unwraps ErrorInfoValue to null), including cycle nodes whose retained
        // cycle edges would violate the DAG invariant of Kahn's algorithm. `keepsEdges()` is false
        // under --notrack_incremental_state (NonIncrementalInMemoryNodeEntry), where direct deps
        // are discarded on completion and getCompressedDirectDepsForDoneEntry() throws.
        if (value != null && entry.keepsEdges()) {
          chunk.entries[id & CHUNK_MASK] = entry;
          visitCompressedDeps(parent, entry.getCompressedDirectDepsForDoneEntry());
        }
      }
    }

    /**
     * Visits the deps in a compressed {@code GroupedDeps}: either an {@code Object[]} of {@link
     * SkyKey}s and {@code Integer} group markers, a single bare {@link SkyKey}, or an opaque empty
     * marker.
     */
    private void visitCompressedDeps(CountedCompleter<?> parent, Object compressed) {
      if (!(compressed instanceof Object[] depArray)) {
        ActionLookupKey depKey = asRelevantKey(compressed);
        if (depKey != null) {
          nodes.incrementIndegree(generateIdAndEnqueueDirectDeps(depKey, parent));
        }
        return;
      }
      // Look all dependencies up first, and only then bump their in-degrees: the out-of-order
      // engine can overlap the cache misses of the hash-table lookups across iterations, whereas
      // the atomic increments in the second loop are full fences that would otherwise serialize
      // them.
      int[] depIds = new int[depArray.length];
      int n = 0;
      for (Object elem : depArray) {
        ActionLookupKey depKey = asRelevantKey(elem);
        if (depKey != null) {
          depIds[n++] = generateIdAndEnqueueDirectDeps(depKey, parent);
        }
      }
      for (int i = 0; i < n; i++) {
        nodes.incrementIndegree(depIds[i]);
      }
    }
  }

  /** Fork/join task that visits one node during {@link Discovery}. */
  private static final class VisitTask extends CountedCompleter<Void> {
    private final Discovery discovery;
    private final ActionLookupKey key;
    private final int id;

    VisitTask(CountedCompleter<?> parent, Discovery discovery, ActionLookupKey key, int id) {
      super(parent);
      this.discovery = discovery;
      this.key = key;
      this.id = id;
    }

    @Override
    public void compute() {
      discovery.visit(this, key, id);
      tryComplete();
    }
  }

  /**
   * Runs the discovery pass from {@code rootKeys}, filling {@code rootIds} with their IDs, and
   * returns the discovered nodes as a graph.
   */
  private static NodeGraph discover(
      InMemoryGraph graph, ConfiguredTargetKey[] rootKeys, int[] rootIds, ForkJoinPool pool)
      throws InterruptedException {
    Discovery discovery = new Discovery(graph);
    CountedCompleter<Void> discoveryRoot =
        new CountedCompleter<Void>() {
          @Override
          public void compute() {
            for (int i = 0; i < rootKeys.length; i++) {
              rootIds[i] = discovery.generateIdAndEnqueueDirectDeps(rootKeys[i], this);
            }
            tryComplete();
          }
        };
    ForkJoinUtil.run(pool, discoveryRoot);
    return new NodeGraph(discovery.ids, discovery.nodes, discovery.nextId.get());
  }

  /**
   * The discovered nodes as a {@link TransitiveClosureCounter.Graph}. Dependencies are read from
   * the Skyframe node entries and mapped to IDs with the key to ID map of the discovery pass.
   */
  private static final class NodeGraph implements TransitiveClosureCounter.Graph {
    private final ObjectToIntMap<ActionLookupKey> ids;
    private final Nodes nodes;
    private final int nodeCount;

    NodeGraph(ObjectToIntMap<ActionLookupKey> ids, Nodes nodes, int nodeCount) {
      this.ids = ids;
      this.nodes = nodes;
      this.nodeCount = nodeCount;
    }

    @Override
    public int nodeCount() {
      return nodeCount;
    }

    @Override
    public int indegree(int id) {
      return nodes.chunk(id).indegrees[id & CHUNK_MASK];
    }

    @Override
    public void read(int id, TransitiveClosureCounter.NodeView out) {
      Chunk chunk = nodes.chunk(id);
      int index = id & CHUNK_MASK;
      out.actions = chunk.actionsPlusOne[index] - 1;
      out.size = 0;
      InMemoryNodeEntry entry = chunk.entries[index];
      if (entry == null) {
        return;
      }
      // The same deps, in the same order, as visited by Discovery.visitCompressedDeps(), so that
      // they match the in-degrees.
      Object compressed = entry.getCompressedDirectDepsForDoneEntry();
      if (!(compressed instanceof Object[] depArray)) {
        ActionLookupKey depKey = asRelevantKey(compressed);
        if (depKey != null) {
          out.deps = ensureCapacity(out, 1);
          out.deps[out.size++] = idOf(depKey);
        }
        return;
      }
      int[] deps = ensureCapacity(out, depArray.length);
      int n = 0;
      for (Object elem : depArray) {
        ActionLookupKey depKey = asRelevantKey(elem);
        if (depKey != null) {
          deps[n++] = idOf(depKey);
        }
      }
      out.deps = deps;
      out.size = n;
    }

    private int idOf(ActionLookupKey key) {
      // No lock needed: the map is complete and no longer modified once discovery is done, which
      // happens-before propagation starts (discover() waits for the discovery tasks to complete,
      // and the propagation tasks are forked afterwards). This is called once per edge per pass,
      // from all propagation threads, so taking the lock would only add contention.
      int id = ids.getUnsynchronized(key);
      checkState(id >= 0, "%s was not discovered", key);
      return id;
    }

    /** Returns a per-thread scratch array of at least {@code size} ints, kept in {@code out}. */
    private static int[] ensureCapacity(TransitiveClosureCounter.NodeView out, int size) {
      int[] scratch = out.scratch instanceof int[] array ? array : new int[0];
      if (scratch.length < size) {
        scratch = new int[Math.max(size, 2 * scratch.length)];
        out.scratch = scratch;
      }
      return scratch;
    }
  }

  public static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots) throws InterruptedException {
    return count(graph, roots, TransitiveClosureCounter.MAX_MASK_BYTES);
  }

  @VisibleForTesting
  static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots, long maxMaskBytes)
      throws InterruptedException {
    if (roots.isEmpty()) {
      return ImmutableMap.of();
    }
    return ForkJoinUtil.withPool(
        "transitive-target-counter", pool -> count(graph, roots, pool, maxMaskBytes));
  }

  private static ImmutableMap<ConfiguredTargetKey, Counts> count(
      InMemoryGraph graph, Set<ConfiguredTargetKey> roots, ForkJoinPool pool, long maxMaskBytes)
      throws InterruptedException {
    ConfiguredTargetKey[] rootKeys = roots.toArray(new ConfiguredTargetKey[0]);
    int numRoots = rootKeys.length;

    // 1. Discovery, assigns dense integer IDs to the nodes of the closures.
    int[] rootIds = new int[numRoots];
    NodeGraph nodeGraph;
    try (SilentCloseable c =
        Profiler.instance().profile("SkyframeTransitiveTargetCounter.discovery")) {
      nodeGraph = discover(graph, rootKeys, rootIds, pool);
    }

    // 2. Propagation.
    long[] totals;
    try (SilentCloseable c =
        Profiler.instance().profile("SkyframeTransitiveTargetCounter.propagation")) {
      totals = TransitiveClosureCounter.count(nodeGraph, rootIds, pool, maxMaskBytes);
    }

    ImmutableMap.Builder<ConfiguredTargetKey, Counts> result =
        ImmutableMap.builderWithExpectedSize(numRoots);
    for (int i = 0; i < numRoots; i++) {
      result.put(rootKeys[i], new Counts(totals[i], totals[numRoots + i]));
    }
    return result.buildOrThrow();
  }
}
