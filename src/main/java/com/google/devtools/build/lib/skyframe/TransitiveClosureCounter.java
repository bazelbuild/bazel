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

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.concurrent.ForkJoinUtil;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountedCompleter;
import java.util.concurrent.ForkJoinPool;
import java.util.stream.IntStream;
import javax.annotation.Nullable;

/**
 * Counts the analysis nodes and actions in the transitive closure of each of a set of roots in a
 * DAG of analysis nodes with dense integer IDs, given as a {@link Graph}.
 *
 * <p>This algorithm traverses the DAG using Kahn's topological sort algorithm to discover which
 * roots can reach a given node. Reachability is propagated top-down via a bitmask (one bit for each
 * root) using simple bitwise OR. If the bitmasks of all roots do not fit into a memory budget, the
 * roots are counted in several passes over subsets of them.
 *
 * <p>We use a fork/join traversal algorithm with one task per node. A node is processed after its
 * parents as per Kahn's algorithm. When processing a node, we add its counts to every root that can
 * reach it according to its bitmask. We maintain a per-thread accumulator for each root node, which
 * are then summed up at the end.
 */
public final class TransitiveClosureCounter {

  /** The transitive analysis node and action counts of a root. */
  public record Counts(long nodes, long actions) {}

  /** The default upper bound on the size of the bitmasks of a pass, to bound memory. */
  public static final long MAX_MASK_BYTES = 1L << 27;

  /** The exclusive upper bound of the {@code maxMaskBytes} argument of {@link #count}. */
  private static final long MAX_MASK_BYTES_LIMIT = 512L << 20;

  /**
   * A DAG with nodes {@code 0 .. nodeCount() - 1}. Must be safe to read from multiple threads at
   * once.
   */
  public interface Graph {
    int nodeCount();

    /** The number of edges pointing to {@code id}. */
    int indegree(int id);

    /**
     * Fills {@code out} with the action count of {@code id} (-1 if it is not counted) and its
     * dependencies. Each thread uses its own {@code out}, so implementations can keep scratch space
     * in {@link NodeView#scratch}.
     */
    void read(int id, NodeView out);
  }

  /** A view of a node of a {@link Graph}, reused for many nodes by the same thread. */
  public static final class NodeView {
    /** The number of actions of the node, or -1 if the node is not counted. */
    public int actions;

    /** The dependencies of the node are {@code deps[0 .. size)}. */
    public int[] deps = NO_DEPS;

    public int size;

    /** Scratch space owned by the {@link Graph} implementation. */
    @Nullable public Object scratch;
  }

  /** Atomic element access on {@link Propagation#state} ({@code long[]}). */
  private static final VarHandle LONG_ARRAY = MethodHandles.arrayElementVarHandle(long[].class);

  private static final int[] NO_DEPS = new int[0];

  private TransitiveClosureCounter() {}

  /**
   * Returns whether {@code key} is the key of a node of the counted analysis graph, that is, a node
   * that can be or transitively reach a {@link
   * com.google.devtools.build.lib.analysis.RuleConfiguredObjectValue}: a configured target with a
   * configuration or an aspect.
   *
   * <p>Non-configurable {@link ConfiguredTargetKey}s ({@code configurationKey == null}, i.e. {@code
   * InputFile}, {@code PackageGroup}, {@code EnvironmentGroup}) always produce {@code
   * NonRuleConfiguredTargetValue} and only depend on visibility {@code PackageGroup}s, so they can
   * never be or reach a {@code RuleConfiguredObjectValue}.
   */
  public static boolean isRelevant(@Nullable Object key) {
    if (key instanceof ConfiguredTargetKey ctk) {
      return ctk.getConfigurationKey() != null;
    }
    return key instanceof AspectKeyCreator.AspectBaseKey;
  }

  /**
   * Returns the transitive node and action counts of each of {@code rootIds} as {@code
   * [nodes[0..rootIds.length), actions[0..rootIds.length)]}.
   */
  public static long[] count(Graph graph, int[] rootIds, ForkJoinPool pool, long maxMaskBytes)
      throws InterruptedException {
    // Sanity check: the bitmasks are allocated in a single array, so a budget this large is most
    // likely a mistake.
    checkArgument(
        maxMaskBytes < MAX_MASK_BYTES_LIMIT,
        "maxMaskBytes must be less than %s, was %s",
        MAX_MASK_BYTES_LIMIT,
        maxMaskBytes);
    int numRoots = rootIds.length;
    long[] totals = new long[2 * numRoots];
    int nodeCount = graph.nodeCount();
    // The number of bitmask words per node in a pass. Each node needs one bit per root of the pass,
    // so all roots need wordsForAllRoots 64-bit words per node, and the bitmasks of a pass with w
    // words take 8 * w * nodeCount bytes. So a pass can have at most maxMaskBytes / (8 *
    // nodeCount) words, but never more than needed for all roots and at least one, so that every
    // pass makes progress even if the budget is too small. Math.max(1, nodeCount) avoids dividing
    // by zero for an empty graph.
    int wordsForAllRoots = (numRoots + 63) / 64;
    int maxWordsPerPass =
        (int) Math.max(1, Math.min(wordsForAllRoots, maxMaskBytes / (8L * Math.max(1, nodeCount))));
    for (int iFirstRoot = 0; iFirstRoot < numRoots; iFirstRoot += 64 * maxWordsPerPass) {
      int iLastRoot = Math.min(numRoots, iFirstRoot + 64 * maxWordsPerPass);
      int passRoots = iLastRoot - iFirstRoot;
      Propagation propagation = new Propagation(graph, iFirstRoot, iLastRoot);
      ForkJoinUtil.run(pool, propagation.createMainTask(rootIds));
      // Sum up per-thread accumulators.
      long visitedNodes = 0;
      for (long[] acc : propagation.accumulators) {
        for (int i = 0; i < passRoots; i++) {
          totals[iFirstRoot + i] += acc[i];
          totals[numRoots + iFirstRoot + i] += acc[passRoots + i];
        }
        visitedNodes += acc[2 * passRoots];
      }
      // Kahn's algorithm visits a node once all of its parents were visited, so the nodes on a
      // cycle and everything below them are never visited. The same happens if indegree()
      // overstates the number of parents of a node. Either way, the counts would be too low.
      checkState(
          visitedNodes == nodeCount,
          "Visited only %s of %s nodes: the graph has a cycle or inconsistent in-degrees",
          visitedNodes,
          nodeCount);
    }
    return totals;
  }

  /**
   * Shared state and traversal logic of one propagation pass over the root range {@code
   * [iFirstRoot, iLastRoot)}.
   */
  private static final class Propagation {
    private final Graph graph;
    private final int iFirstRoot;
    private final int iLastRoot;
    private final int words;

    /**
     * Number of {@code long} slots per node in {@link #state}: {@link #words} bitmask words plus 1
     * slot for the remaining in-degree.
     */
    private final int stride;

    //   |<- node 0 (stride = words + 1) ->|<- node 1 (stride = words + 1) ->| ...
    //   +---------+-----+-----------+-----+---------+-----+-----------+-----+
    //   | mask[0] | ... | mask[w-1] |indeg| mask[0] | ... | mask[w-1] |indeg| ...
    //   +---------+-----+-----------+-----+---------+-----+-----------+-----+
    /**
     * Per node: {@code words} mask words followed by the remaining in-degree. Bit {@code b} of a
     * node's mask (bit {@code b % 64} of word {@code b / 64}) is set if root {@code iFirstRoot + b}
     * reaches the node.
     */
    private final long[] state;

    /**
     * Thread-local {@code [nodes..., actions..., visited nodes]} accumulators: one pair of slots
     * per root of this pass (index {@code i - iFirstRoot}) and the number of nodes the thread
     * visited.
     */
    private final ThreadLocal<long[]> accumulator;

    private final ThreadLocal<NodeView> nodeView = ThreadLocal.withInitial(NodeView::new);

    final ConcurrentLinkedQueue<long[]> accumulators = new ConcurrentLinkedQueue<>();

    Propagation(Graph graph, int iFirstRoot, int iLastRoot) {
      this.graph = graph;
      this.iFirstRoot = iFirstRoot;
      this.iLastRoot = iLastRoot;
      this.words = (iLastRoot - iFirstRoot + 63) / 64;
      // + 1 for the remaining in-degree stored after each node's bitmask words.
      this.stride = words + 1;
      this.state = new long[graph.nodeCount() * stride];
      this.accumulator =
          ThreadLocal.withInitial(() -> {
            long[] acc = new long[2 * (iLastRoot - iFirstRoot) + 1];
            accumulators.add(acc);
            return acc;
          });
    }

    CountedCompleter<Void> createMainTask(int[] rootIds) {
      return new CountedCompleter<Void>() {
        @Override
        public void compute() {
          start(this, rootIds);
          tryComplete();
        }
      };
    }

    /**
     * Initializes {@link #state} and forks {@link PropagateTask}s for all nodes without parents.
     */
    private void start(CountedCompleter<?> parent, int[] rootIds) {
      int nodeCount = graph.nodeCount();
      // Runs on a pool worker, so the parallel stream is executed by this pool too.
      IntStream.range(0, nodeCount)
          .parallel()
          .forEach(id -> state[id * stride + words] = graph.indegree(id));
      // Mark each root as being reachable from itself.
      for (int i = iFirstRoot; i < iLastRoot; i++) {
        int bit = i - iFirstRoot;
        state[rootIds[i] * stride + bit / 64] |= 1L << (bit % 64);
      }
      // Start every node without parents, not just the roots in [iFirstRoot, iLastRoot), so that
      // Kahn's algorithm reaches roots whose parents are roots from other passes and nodes whose
      // parents no root reaches. This must check the in-degree in the graph, not the remaining one
      // in `state`: tasks forked here already run and decrement remaining in-degrees, so a node
      // they make ready would be started twice.
      for (int id = 0; id < nodeCount; id++) {
        if (graph.indegree(id) == 0) {
          parent.addToPendingCount(1);
          new PropagateTask(parent, this, id).fork();
        }
      }
    }

    /**
     * Adds this node and its actions to the accumulator, ORs its bitmask into that of its direct
     * deps and enqueues its ready children.
     */
    void propagate(CountedCompleter<?> parent, int id) {
      NodeView node = nodeView.get();
      graph.read(id, node);
      int base = id * stride;
      long[] acc = accumulator.get();
      acc[acc.length - 1]++;

      // If the node is counted, add it and its actions to the accumulators of the roots that reach
      // it. `actions` is -1 for nodes that are not counted (e.g. non-rule configured targets), and
      // nodes with zero actions are still counted.
      if (node.actions >= 0) {
        int actionsBase = (acc.length - 1) / 2;
        for (int w = 0; w < words; w++) {
          long mask = state[base + w];
          int rootBase = w * 64;
          // Iterate over the set bits in `mask`; where a bit is set, update the accumulator
          // accordingly.
          while (mask != 0) {
            int root = rootBase + Long.numberOfTrailingZeros(mask);
            acc[root]++;
            acc[actionsBase + root] += node.actions;
            // Unset the lowest set bit.
            mask &= mask - 1;
          }
        }
      }

      // Iterate over every dep: OR the mask of this node into the mask of the dep and decrement the
      // remaining in-degree of the dep by one. If the dep has no more parents remaining, enqueue
      // it.
      int[] deps = node.deps;
      for (int i = 0; i < node.size; i++) {
        int dep = deps[i];
        int depBase = dep * stride;
        for (int w = 0; w < words; w++) {
          long mask = state[base + w];
          // The OR is atomic (getAndBitwiseOr), so parents ORing into the same dep concurrently
          // never lose each other's bits. The plain read before it only skips the (fenced) atomic
          // OR if the dep already has all of our bits, which is common when parents share roots.
          // Bits are only ever set, so a stale read just means that we do the atomic OR anyway.
          if (mask != 0) {
            long current = state[depBase + w];
            if ((current | mask) != current) {
              LONG_ARRAY.getAndBitwiseOr(state, depBase + w, mask);
            }
          }
        }
        if ((long) LONG_ARRAY.getAndAdd(state, depBase + words, -1L) == 1L) {
          parent.addToPendingCount(1);
          new PropagateTask(parent, this, dep).fork();
        }
      }
    }
  }

  /** Fork/join task that processes one ready node during {@link Propagation}. */
  private static final class PropagateTask extends CountedCompleter<Void> {
    private final Propagation propagation;
    private final int id;

    PropagateTask(CountedCompleter<?> parent, Propagation propagation, int id) {
      super(parent);
      this.propagation = propagation;
      this.id = id;
    }

    @Override
    public void compute() {
      propagation.propagate(this, id);
      tryComplete();
    }
  }
}
