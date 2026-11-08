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
package com.google.devtools.build.lib.query2.cquery;

import com.google.devtools.build.lib.query2.query.hash.TransitiveHashes;
import com.google.devtools.build.lib.util.Fingerprint;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/** Hashes the formatter's closed graph of interned configured nodes in place. */
final class CompactConfiguredHashes {
  private CompactConfiguredHashes() {}

  // Claim multiple vertices per task to amortize dispatch. Narrow levels run inline.
  private static final int MIN_CHUNK = 64;
  private static final int MAX_INDEXED_LEVELS = 1 << 15;

  // Every listed dependency is an interned member of the same graph. A missing direct hash
  // or failed dependency walk poisons dependents, as in the keep-going formatter.
  static class Node<N extends Node<N>> {
    final int ordinal;
    // Populated before any worker starts sorting or combining. A parallel pass can fill all stable
    // names after interning.
    String name;
    List<N> deps = List.of();
    @Nullable byte[] hash;

    Node(int ordinal, String name) {
      this.ordinal = ordinal;
      this.name = name;
    }
  }

  static <N extends Node<N>> void combine(List<N> nodes, Set<N> failed) {
    int[] order = topologicalOrder(nodes);
    if (order == null) {
      // Some projected configured graphs may contain cycles. Preserve the original SCC
      // implementation, traversal and fingerprint order, using the untouched direct hashes.
      Map<N, TransitiveHashes.Node<N>> graph = HashMap.newHashMap(nodes.size());
      for (N node : nodes) {
        if (node.hash != null) {
          graph.put(node, new TransitiveHashes.Node<>(node.hash, node.deps));
        }
      }
      Map<N, byte[]> hashes =
          TransitiveHashes.combine(
              graph,
              (left, right) -> {
                int result = left.name.compareTo(right.name);
                if (result == 0 && left != right) {
                  throw new UncheckedIOException(
                      new IOException(
                          "Distinct members of a configured hash cycle have the same identity"));
                }
                return result;
              },
              node -> node.name,
              failed);
      for (N node : nodes) {
        node.hash = hashes.get(node);
      }
      return;
    }
    for (int ordinal : order) {
      N node = nodes.get(ordinal);
      if (node.hash == null || failed.contains(node)) {
        node.hash = null;
        continue;
      }
      // Dependencies have already replaced their direct hash with the final result.
      Fingerprint fp = new Fingerprint().addBytes(node.hash);
      boolean complete = true;
      for (N dep : node.deps) {
        if (dep.hash == null) {
          complete = false;
          break;
        }
        fp.addString(dep.name).addString("").addBytes(dep.hash);
      }
      node.hash = complete ? fp.digestAndReset() : null;
    }
  }

  /** Validation-first DAG combine. All tasks at one dependency height finish before the next. */
  static <N extends Node<N>> void combineParallel(List<N> nodes, Set<N> failed, int workers)
      throws InterruptedException {
    Levels levels = dependencyLevels(nodes);
    if (levels == null) {
      // Validation happens before modifying any hash, including on cycles and extreme depth.
      combine(nodes, failed);
      return;
    }
    int budget = Math.max(1, Math.min(workers, nodes.size()));
    var pool = budget > 1 ? Executors.newFixedThreadPool(budget) : null;
    var local = ThreadLocal.withInitial(Fingerprint::new);
    Fingerprint serial = new Fingerprint();
    try {
      for (int h = 0; h + 1 < levels.starts.length; h++) {
        int first = levels.starts[h];
        int end = levels.starts[h + 1];
        int width = end - first;
        int tasks = Math.min(budget, (width + MIN_CHUNK - 1) / MIN_CHUNK);
        if (tasks <= 1) {
          combineRange(nodes, failed, levels.ordinals, first, end, serial);
          continue;
        }
        var next = new AtomicInteger(first);
        var calls = new ArrayList<Callable<Void>>(tasks);
        for (int j = 0; j < tasks; j++) {
          calls.add(
              () -> {
                Fingerprint fp = local.get();
                for (int i; (i = next.getAndAdd(MIN_CHUNK)) < end; ) {
                  if (Thread.currentThread().isInterrupted()) {
                    throw new InterruptedException();
                  }
                  combineRange(nodes, failed, levels.ordinals, i, Math.min(i + MIN_CHUNK, end), fp);
                }
                return null;
              });
        }
        for (var future : pool.invokeAll(calls)) {
          try {
            future.get();
          } catch (ExecutionException e) {
            if (e.getCause() instanceof InterruptedException interrupted) {
              throw interrupted;
            }
            if (e.getCause() instanceof RuntimeException runtime) {
              throw runtime;
            }
            if (e.getCause() instanceof Error error) {
              throw error;
            }
            throw new IllegalStateException(e.getCause());
          }
        }
      }
    } finally {
      if (pool != null) {
        pool.shutdownNow();
        pool.close();
      }
    }
  }

  private static <N extends Node<N>> void combineRange(
      List<N> nodes, Set<N> failed, int[] ordinals, int first, int end, Fingerprint fp) {
    for (int i = first; i < end; i++) {
      N node = nodes.get(ordinals[i]);
      if (node.hash == null || failed.contains(node)) {
        node.hash = null;
        continue;
      }
      fp.addBytes(node.hash);
      boolean complete = true;
      // Keep the original sorted iteration and byte framing exactly. addString already reads
      // Bazel's internal string bytes. Re-encoding or pre-hashing names would be wasteful.
      for (N dep : node.deps) {
        if (dep.hash == null) {
          complete = false;
          break;
        }
        fp.addString(dep.name).addString("").addBytes(dep.hash);
      }
      // A poisoned vertex can have written a prefix. Reset the reusable fingerprint in both cases.
      byte[] total = fp.digestAndReset();
      node.hash = complete ? total : null;
    }
  }

  private record Levels(int[] starts, int[] ordinals) {}

  /** Dependency height and level grouping share the same V-sized scratch as the original DFS. */
  @Nullable
  private static <N extends Node<N>> Levels dependencyLevels(List<N> nodes) {
    byte[] state = new byte[nodes.size()];
    int[] cursor = new int[nodes.size()];
    int[] height = new int[nodes.size()];
    int[] counts = new int[32];
    int max = -1;
    ArrayDeque<N> stack = new ArrayDeque<>();
    for (N root : nodes) {
      if (state[root.ordinal] != 0) {
        continue;
      }
      stack.push(root);
      state[root.ordinal] = 1;
      while (!stack.isEmpty()) {
        N node = stack.peek();
        int i = node.ordinal;
        if (node.hash != null && cursor[i] < node.deps.size()) {
          N child = node.deps.get(cursor[i]++);
          int c = child.ordinal;
          if (state[c] == 1) {
            return null;
          }
          if (state[c] == 0) {
            state[c] = 1;
            stack.push(child);
          } else {
            height[i] = Math.max(height[i], height[c] + 1);
          }
          continue;
        }
        stack.pop();
        state[i] = 2;
        int h = height[i];
        // Bound the height index; very deep models use the existing iterative implementation.
        if (h >= MAX_INDEXED_LEVELS) {
          return null;
        }
        if (h >= counts.length) {
          counts = Arrays.copyOf(counts, counts.length * 2);
        }
        counts[h]++;
        max = Math.max(max, h);
        if (!stack.isEmpty()) {
          int parent = stack.peek().ordinal;
          height[parent] = Math.max(height[parent], h + 1);
        }
      }
    }
    int[] starts = new int[max + 2];
    for (int h = 0; h <= max; h++) {
      starts[h + 1] = starts[h] + counts[h];
      counts[h] = starts[h];
    }
    // DFS cursors are dead now, so recycle that array for the output ordinals.
    for (int i = 0; i < height.length; i++) {
      cursor[counts[height[i]]++] = i;
    }
    return new Levels(starts, cursor);
  }

  /** Returns dependency-first ordinals, or null upon finding a cycle, before modifying hashes. */
  @Nullable
  private static <N extends Node<N>> int[] topologicalOrder(List<N> nodes) {
    byte[] state = new byte[nodes.size()]; // 0 = unseen, 1 = active, 2 = completed
    int[] nextDep = new int[nodes.size()];
    int[] order = new int[nodes.size()];
    int count = 0;
    ArrayDeque<N> stack = new ArrayDeque<>();
    for (N root : nodes) {
      if (state[root.ordinal] != 0) {
        continue;
      }
      stack.push(root);
      state[root.ordinal] = 1;
      while (!stack.isEmpty()) {
        N current = stack.peek();
        int index = current.ordinal;
        if (current.hash != null && nextDep[index] < current.deps.size()) {
          N dep = current.deps.get(nextDep[index]++);
          if (state[dep.ordinal] == 1) {
            return null;
          }
          if (state[dep.ordinal] == 0) {
            state[dep.ordinal] = 1;
            stack.push(dep);
          }
        } else {
          stack.pop();
          state[index] = 2;
          order[count++] = index;
        }
      }
    }
    return order;
  }
}
