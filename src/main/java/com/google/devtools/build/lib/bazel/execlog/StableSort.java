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
package com.google.devtools.build.lib.bazel.execlog;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Multimap;
import com.google.common.collect.MultimapBuilder;
import com.google.devtools.build.lib.exec.Protos.File;
import com.google.devtools.build.lib.exec.Protos.SpawnExec;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.util.io.MessageOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

/**
 * A Utility to sort the SpawnExec log in a way that is reproducible across nondeterministic Bazel
 * runs.
 *
 * <p>This is needed to allow textual diff comparisons of resultant logs.
 */
public final class StableSort {
  private static ImmutableList<SpawnExec> read(InputStream in) throws IOException {
    ImmutableList.Builder<SpawnExec> result = ImmutableList.builder();
    while (in.available() > 0) {
      SpawnExec ex = SpawnExec.parseDelimitedFrom(in);
      result.add(ex);
    }
    return result.build();
  }

  /**
   * Reads binary SpawnLog protos from the InputStream, sorts them and outputs to the
   * MessageOutputStream.
   *
   * <p>The sorting is done according to the following rules: - If some output of action A appears
   * as an input to action B, A will appear before B. - When not constrained by (transitive) 1., any
   * action whose name of the first output is lexicographically smaller would appear earlier.
   *
   * <p>We assume that in the InputStream, at most one SpawnExec declares a given file as its
   * output. We assume that there are no cyclic dependencies.
   */
  public static void stableSort(InputStream in, MessageOutputStream out) throws IOException {
    try (SilentCloseable c = Profiler.instance().profile("stableSort")) {
      ImmutableList<SpawnExec> inputs;
      try (SilentCloseable c2 = Profiler.instance().profile("stableSort/read")) {
        inputs = read(in);
      }
      stableSort(inputs, out);
    }
  }

  private static void stableSort(List<SpawnExec> inputs, MessageOutputStream out)
      throws IOException {
    // A map from each output to a SpawnExec that produced it
    Multimap<String, SpawnExec> outputProducer =
        MultimapBuilder.hashKeys(inputs.size()).arrayListValues().build();

    for (SpawnExec ex : inputs) {
      for (File output : ex.getActualOutputsList()) {
        String name = output.getPath();
        outputProducer.put(name, ex);
      }
    }

    // A spawnExec a blocks b if a produces an output consumed by b
    Multimap<SpawnExec, SpawnExec> blockedBy = MultimapBuilder.hashKeys().arrayListValues().build();
    Multimap<SpawnExec, SpawnExec> blocking = MultimapBuilder.hashKeys().arrayListValues().build();

    for (SpawnExec ex : inputs) {
      for (File s : ex.getInputsList()) {
        if (outputProducer.containsKey(s.getPath())) {
          for (SpawnExec blocker : outputProducer.get(s.getPath())) {
            blockedBy.put(ex, blocker);
            blocking.put(blocker, ex);
          }
        }
      }
    }

    // This is a queue of all spawnExecs that are not blocked by future spawnExecs
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
                  stripped.setProgressMessage(o.getProgressMessage());
                  stripped.setMnemonic(o.getMnemonic());

                  return "2_" + stripped.build();
                }));
    for (SpawnExec ex : inputs) {
      if (!blockedBy.containsKey(ex)) {
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
