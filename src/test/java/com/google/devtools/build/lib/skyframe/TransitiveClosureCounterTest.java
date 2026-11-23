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

import static com.google.common.truth.Truth.assertThat;
import static org.junit.Assert.assertThrows;

import com.google.devtools.build.lib.analysis.config.BuildOptions;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.concurrent.ForkJoinUtil;
import com.google.devtools.build.lib.skyframe.config.BuildConfigurationKey;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

/** Unit tests for {@link TransitiveClosureCounter}. */
@RunWith(JUnit4.class)
public final class TransitiveClosureCounterTest {

  /** A graph given as action counts and dep lists. */
  private static final class ArrayGraph implements TransitiveClosureCounter.Graph {
    private final int[] actions;
    private final int[][] deps;
    private final int[] indegrees;

    ArrayGraph(int[] actions, int[][] deps) {
      this.actions = actions;
      this.deps = deps;
      this.indegrees = new int[actions.length];
      for (int[] nodeDeps : deps) {
        for (int dep : nodeDeps) {
          indegrees[dep]++;
        }
      }
    }

    @Override
    public int nodeCount() {
      return actions.length;
    }

    @Override
    public int indegree(int id) {
      return indegrees[id];
    }

    @Override
    public void read(int id, TransitiveClosureCounter.NodeView out) {
      out.actions = actions[id];
      out.deps = deps[id];
      out.size = deps[id].length;
    }
  }

  private static long[] count(ArrayGraph graph, int[] rootIds, long maxMaskBytes)
      throws InterruptedException {
    return ForkJoinUtil.withPool(
        "test", pool -> TransitiveClosureCounter.count(graph, rootIds, pool, maxMaskBytes));
  }

  @Test
  public void diamond_countsSharedDependencyOnce() throws Exception {
    // 0 -> {1, 2}, 1 -> 3, 2 -> 3.
    ArrayGraph graph = new ArrayGraph(new int[] {1, 2, 3, 4}, new int[][] {{1, 2}, {3}, {3}, {}});

    assertThat(count(graph, new int[] {0, 1}, 1L << 20))
        .asList()
        .containsExactly(4L, 2L, 10L, 6L)
        .inOrder();
  }

  @Test
  public void uncountedNodes_areTraversedButNotCounted() throws Exception {
    // 0 -> 1 -> 2, where 1 is not counted.
    ArrayGraph graph = new ArrayGraph(new int[] {1, -1, 5}, new int[][] {{1}, {2}, {}});

    assertThat(count(graph, new int[] {0}, 1L << 20)).asList().containsExactly(2L, 6L).inOrder();
  }

  @Test
  public void nodesWithoutActions_areCounted() throws Exception {
    // 0 -> 1 -> 2, where 1 has no actions but is counted.
    ArrayGraph graph = new ArrayGraph(new int[] {1, 0, 5}, new int[][] {{1}, {2}, {}});

    assertThat(count(graph, new int[] {0}, 1L << 20)).asList().containsExactly(3L, 6L).inOrder();
  }

  @Test
  public void maxMaskBytesTooLarge_throws() throws Exception {
    ArrayGraph graph = new ArrayGraph(new int[] {1}, new int[][] {{}});

    assertThrows(IllegalArgumentException.class, () -> count(graph, new int[] {0}, 512L << 20));
  }

  @Test
  public void cycle_throws() throws Exception {
    // 0 -> 1 -> 2 -> 1 -> ... and 2 -> 3. Nodes 1, 2 and 3 are never visited.
    ArrayGraph graph = new ArrayGraph(new int[] {1, 2, 3, 4}, new int[][] {{1}, {2}, {1, 3}, {}});

    IllegalStateException e =
        assertThrows(IllegalStateException.class, () -> count(graph, new int[] {0}, 1L << 20));
    assertThat(e).hasMessageThat().contains("Visited only 1 of 4 nodes");
  }

  @Test
  public void nodeWithoutParentsThatIsNotARoot_doesNotStallTraversal() throws Exception {
    // 0 -> 2 and 1 -> 2; only 0 is a root, but 2 has a parent that no root reaches.
    ArrayGraph graph = new ArrayGraph(new int[] {1, 10, 100}, new int[][] {{2}, {2}, {}});

    assertThat(count(graph, new int[] {0}, 1L << 20)).asList().containsExactly(2L, 101L).inOrder();
  }

  @Test
  public void manyNodesWithoutParents_eachNodeProcessedOnce() throws Exception {
    // Nodes [0, n) have no parents and each depends on n + i. Propagation of the first ones can
    // make the latter ones ready while nodes without parents are still being started; they must
    // not be started a second time.
    int n = 20_000;
    int[] actions = new int[2 * n];
    int[][] deps = new int[2 * n][];
    int[] rootIds = new int[64];
    for (int i = 0; i < n; i++) {
      actions[i] = 1;
      deps[i] = new int[] {n + i};
      actions[n + i] = 1;
      deps[n + i] = new int[0];
    }
    for (int r = 0; r < rootIds.length; r++) {
      rootIds[r] = r * (n / rootIds.length);
    }
    ArrayGraph graph = new ArrayGraph(actions, deps);

    for (int attempt = 0; attempt < 10; attempt++) {
      long[] totals = count(graph, rootIds, 1L << 20);
      for (int r = 0; r < rootIds.length; r++) {
        assertThat(totals[r]).isEqualTo(2);
        assertThat(totals[rootIds.length + r]).isEqualTo(2);
      }
    }
  }

  @Test
  public void rootDependingOnRootInAnotherPass() throws Exception {
    // A chain where node i depends on node i + 1; every node is a root. A tiny mask budget forces
    // one pass per 64 roots, so roots are reached through parents counted in other passes.
    int n = 200;
    int[] actions = new int[n];
    int[][] deps = new int[n][];
    int[] rootIds = new int[n];
    for (int i = 0; i < n; i++) {
      actions[i] = 1;
      deps[i] = i == n - 1 ? new int[0] : new int[] {i + 1};
      rootIds[i] = i;
    }

    long[] totals = count(new ArrayGraph(actions, deps), rootIds, /* maxMaskBytes= */ 1);

    for (int i = 0; i < n; i++) {
      assertThat(totals[i]).isEqualTo(n - i);
      assertThat(totals[n + i]).isEqualTo(n - i);
    }
  }

  @Test
  public void randomDag_matchesNaiveTraversal() throws Exception {
    int n = 5_000;
    Random random = new Random(42);
    int[] actions = new int[n];
    int[][] deps = new int[n][];
    for (int i = 0; i < n; i++) {
      actions[i] = random.nextInt(5) == 0 ? -1 : random.nextInt(10);
      // Deps have higher IDs, so the graph is a DAG.
      Set<Integer> depSet = new HashSet<>();
      int degree = i == n - 1 ? 0 : random.nextInt(Math.min(n - 1 - i, 8) + 1);
      for (int d = 0; d < degree; d++) {
        depSet.add(i + 1 + random.nextInt(Math.min(n - 1 - i, 100)));
      }
      deps[i] = depSet.stream().mapToInt(Integer::intValue).toArray();
    }
    List<Integer> roots = new ArrayList<>();
    for (int i = 0; i < n; i += 37) {
      roots.add(i);
    }
    int[] rootIds = roots.stream().mapToInt(Integer::intValue).toArray();

    long[] totals = count(new ArrayGraph(actions, deps), rootIds, /* maxMaskBytes= */ 8 * n);

    for (int r = 0; r < rootIds.length; r++) {
      long nodes = 0;
      long actionCount = 0;
      boolean[] seen = new boolean[n];
      Deque<Integer> stack = new ArrayDeque<>();
      stack.push(rootIds[r]);
      seen[rootIds[r]] = true;
      while (!stack.isEmpty()) {
        int id = stack.pop();
        if (actions[id] >= 0) {
          nodes++;
          actionCount += actions[id];
        }
        for (int dep : deps[id]) {
          if (!seen[dep]) {
            seen[dep] = true;
            stack.push(dep);
          }
        }
      }
      assertThat(totals[r]).isEqualTo(nodes);
      assertThat(totals[rootIds.length + r]).isEqualTo(actionCount);
    }
  }

  @Test
  public void isRelevant() {
    BuildConfigurationKey config = BuildConfigurationKey.create(BuildOptions.builder().build());
    Label label = Label.parseCanonicalUnchecked("//pkg:t");

    assertThat(
            TransitiveClosureCounter.isRelevant(
                ConfiguredTargetKey.builder().setLabel(label).setConfigurationKey(config).build()))
        .isTrue();
    assertThat(
            TransitiveClosureCounter.isRelevant(
                ConfiguredTargetKey.builder().setLabel(label).build()))
        .isFalse();
    assertThat(TransitiveClosureCounter.isRelevant(label)).isFalse();
    assertThat(TransitiveClosureCounter.isRelevant(null)).isFalse();
  }
}
