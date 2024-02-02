// Copyright 2019 The Bazel Authors. All rights reserved.
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
//
package com.google.devtools.build.lib.exec;

import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.common.collect.Sets;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.util.io.MessageInputStream;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.IdentityHashMap;
import java.util.PriorityQueue;
import java.util.Set;

/**
 * A Utility to sort the execution log in a way that is reproducible across nondeterministic Bazel
 * runs.
 *
 * <p>This is needed to allow textual diff comparisons of resultant logs.
 */
public final class StableSort {
  /**
   * Reads {@link SpawnExec} protos from an {@link MessageInputStream}, sorts them, and writes them
   * to a {@link MessageOutputStream}.
   *
   * <p>The sort order has the following properties:
   *
   * <ol>
   *   <li>If an output of spawn A is an input to spawn B, A sorts before B.
   *   <li>When not constrained by the above, spawns sort in lexicographic order of their primary
   *       output path.
   * </ol>
   *
   * <p>Assumes that there are no cyclic dependencies.
   */
  public static void stableSort(
      MessageInputStream<SpawnExec> in, MessageOutputStream<SpawnExec> out) throws IOException {
    try (SilentCloseable c = Profiler.instance().profile("stableSort")) {
      ArrayList<SpawnExec> inputs = new ArrayList<>();

      try (SilentCloseable c2 = Profiler.instance().profile("stableSort/read")) {
        SpawnExec ex;
        while ((ex = in.read()) != null) {
          inputs.add(ex);
        }
      }

      // A map from each output to every spawn that produces it.
      // The same output may be produced by multiple spawns in the case of multiple test attempts.
      Multimap<String, SpawnExec> outputProducer =
          MultimapBuilder.hashKeys(inputs.size()).arrayListValues(1).build();

      for (SpawnExec ex : inputs) {
        for (File output : ex.getActualOutputsList()) {
          String name = output.getPath();
          outputProducer.put(name, ex);
        }
      }

      // A blocks B if A produces an output consumed by B.
      // Use reference equality to avoid expensive comparisons.
      IdentitySetMultimap<SpawnExec, SpawnExec> blockedBy = new IdentitySetMultimap<>();
      IdentitySetMultimap<SpawnExec, SpawnExec> blocking = new IdentitySetMultimap<>();

      // The queue contains all spawns whose blockers have already been emitted.
      PriorityQueue<SpawnExec> queue =
          new PriorityQueue<>(
              Comparator.comparing(
                  o -> {
                    // Sort by comparing the path of the first output. We don't want the sorting to
                    // rely on file hashes because we want the same action graph to be sorted in the
                    // same way regardless of file contents.
                    if (o.getListedOutputsCount() > 0) {
                      return "1_" + o.getListedOutputs(0);
                    }

                    // Get a proto with only stable information from this proto
                    SpawnExec.Builder stripped = SpawnExec.newBuilder();
                    stripped.addAllCommandArgs(o.getCommandArgsList());
                    stripped.addAllEnvironmentVariables(o.getEnvironmentVariablesList());
                    stripped.setPlatform(o.getPlatform());
                    stripped.addAllInputs(o.getInputsList());
                    stripped.setMnemonic(o.getMnemonic());

                    return "2_" + stripped.build();
                  }));

      for (SpawnExec ex : inputs) {
        boolean blocked = false;
        for (File s : ex.getInputsList()) {
          for (SpawnExec blocker : outputProducer.get(s.getPath())) {
            blockedBy.put(ex, blocker);
            blocking.put(blocker, ex);
            blocked = true;
          }
        }
        if (!blocked) {
          queue.add(ex);
        }
      }

      while (!queue.isEmpty()) {
        SpawnExec curr = queue.remove();
        out.write(curr);

        for (SpawnExec blocked : blocking.get(curr)) {
          blockedBy.remove(blocked, curr);
          if (!blockedBy.containsKey(blocked)) {
            queue.add(blocked);
          }
        }
      }
    }
  }

  // A SetMultimap that uses reference equality for keys and values.
  // Implements only the subset of the SetMultimap API needed by stableSort().
  private static class IdentitySetMultimap<K, V> {
    final IdentityHashMap<K, Set<V>> map = new IdentityHashMap<>();

    boolean containsKey(K key) {
      return map.containsKey(key);
    }

    Set<V> get(K key) {
      return map.getOrDefault(key, ImmutableSet.of());
    }

    void put(K key, V value) {
      map.computeIfAbsent(key, k -> Sets.newIdentityHashSet()).add(value);
    }

    void remove(K key, V value) {
      map.compute(
          key,
          (unusedKey, valueSet) -> {
            if (valueSet == null) {
              return null;
            }
            valueSet.remove(value);
            return valueSet.isEmpty() ? null : valueSet;
          });
    }
  }
}
