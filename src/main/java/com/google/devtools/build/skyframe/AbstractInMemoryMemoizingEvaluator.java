// Copyright 2022 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.util.GroupedList;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.errorprone.annotations.ForOverride;
import java.io.PrintStream;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Predicate;
import javax.annotation.Nullable;

/** Partial implementation of {@link MemoizingEvaluator} based on an {@link InMemoryGraph}. */
public abstract class AbstractInMemoryMemoizingEvaluator implements MemoizingEvaluator {

  @ForOverride
  protected abstract InMemoryGraph inMemoryGraph();

  @Override
  public final Map<SkyKey, SkyValue> getValues() {
    return inMemoryGraph().getValues();
  }

  @Override
  public final Set<Entry<SkyKey, InMemoryNodeEntry>> getGraphEntries() {
    return inMemoryGraph().getAllValuesMutable().entrySet();
  }

  @Override
  public final Map<SkyKey, SkyValue> getDoneValues() {
    return inMemoryGraph().getDoneValues();
  }

  private static boolean isDone(@Nullable NodeEntry entry) {
    return entry != null && entry.isDone();
  }

  @Override
  @Nullable
  public final SkyValue getExistingValue(SkyKey key) {
    NodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    try {
      return isDone(entry) ? entry.getValue() : null;
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph does not throw" + key + ", " + entry, e);
    }
  }

  @Override
  @Nullable
  public final ErrorInfo getExistingErrorForTesting(SkyKey key) {
    NodeEntry entry = getExistingEntryAtCurrentlyEvaluatingVersion(key);
    try {
      return isDone(entry) ? entry.getErrorInfo() : null;
    } catch (InterruptedException e) {
      throw new IllegalStateException("InMemoryGraph does not throw" + key + ", " + entry, e);
    }
  }

  @Nullable
  @Override
  public final NodeEntry getExistingEntryAtCurrentlyEvaluatingVersion(SkyKey key) {
    return inMemoryGraph().get(null, Reason.OTHER, key);
  }

  @Override
  public final void dumpSummary(PrintStream out) {
    long nodes = 0;
    long edges = 0;
    for (InMemoryNodeEntry entry : inMemoryGraph().getAllValues().values()) {
      nodes++;
      if (entry.isDone() && entry.keepEdges() != NodeEntry.KeepEdgesPolicy.NONE) {
        edges += Iterables.size(entry.getDirectDeps());
      }
    }
    out.println("Node count: " + nodes);
    out.println("Edge count: " + edges);
  }

  @Override
  public final void dumpCount(PrintStream out) {
    Map<String, AtomicInteger> counter = new HashMap<>();
    for (SkyKey key : inMemoryGraph().getAllValues().keySet()) {
      String mapKey = key.functionName().getName();
      counter.putIfAbsent(mapKey, new AtomicInteger());
      counter.get(mapKey).incrementAndGet();
    }
    for (Entry<String, AtomicInteger> entry : counter.entrySet()) {
      out.println(entry.getKey() + "\t" + entry.getValue()); // \t is spreadsheet-friendly.
    }
  }

  @Override
  public final void dumpDetailed(PrintStream out, Predicate<SkyKey> filter) {
    inMemoryGraph()
        .getAllValues()
        .forEach(
            (key, entry) -> {
              if (!filter.test(key) || !entry.isDone()) {
                return;
              }
              printKey(key, out);
              if (entry.keepEdges() == NodeEntry.KeepEdgesPolicy.NONE) {
                out.println("  (direct deps not stored)");
              } else {
                GroupedList<SkyKey> deps =
                    GroupedList.create(entry.getCompressedDirectDepsForDoneEntry());
                for (int i = 0; i < deps.listSize(); i++) {
                  out.format("  Group %d:\n", i + 1);
                  for (SkyKey dep : deps.get(i)) {
                    out.print("    ");
                    printKey(dep, out);
                  }
                }
              }
              out.println();
            });
  }

  private static void printKey(SkyKey key, PrintStream out) {
    out.format("%s:%s\n", key.functionName(), key.argument().toString().replace('\n', '_'));
  }
}
