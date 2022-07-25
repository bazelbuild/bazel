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
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
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
  public final ConcurrentHashMap<SkyKey, InMemoryNodeEntry> getAllValuesMutable() {
    return inMemoryGraph().getAllValuesMutable();
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
      if (entry.isDone() && entry.keepsEdges()) {
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
  public final void dumpDeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    for (Map.Entry<SkyKey, InMemoryNodeEntry> x : inMemoryGraph().getAllValues().entrySet()) {
      // This can be very long running on large graphs so check for user abort requests.
      if (Thread.interrupted()) {
        out.println("aborting");
        throw new InterruptedException();
      }

      SkyKey key = x.getKey();
      InMemoryNodeEntry entry = x.getValue();
      String canonicalizedKey = canonicalizeKey(key);
      if (!filter.test(canonicalizedKey) || !entry.isDone()) {
        continue;
      }
      out.println(canonicalizedKey);
      if (entry.keepsEdges()) {
        GroupedList<SkyKey> deps = GroupedList.create(entry.getCompressedDirectDepsForDoneEntry());
        for (int i = 0; i < deps.listSize(); i++) {
          out.format("  Group %d:\n", i + 1);
          for (SkyKey dep : deps.get(i)) {
            out.print("    ");
            out.println(canonicalizeKey(dep));
          }
        }
      } else {
        out.println("  (direct deps not stored)");
      }
      out.println();
    }
  }

  @Override
  public final void dumpRdeps(PrintStream out, Predicate<String> filter)
      throws InterruptedException {
    for (Map.Entry<SkyKey, InMemoryNodeEntry> x : inMemoryGraph().getAllValues().entrySet()) {
      // This can be very long running on large graphs so check for user abort requests.
      if (Thread.interrupted()) {
        throw new InterruptedException();
      }

      SkyKey key = x.getKey();
      InMemoryNodeEntry entry = x.getValue();
      String canonicalizedKey = canonicalizeKey(key);
      if (!filter.test(canonicalizedKey) || !entry.isDone()) {
        continue;
      }
      out.println(canonicalizedKey);
      if (entry.keepsEdges()) {
        Collection<SkyKey> rdeps = entry.getReverseDepsForDoneEntry();
        for (SkyKey rdep : rdeps) {
          out.print("    ");
          out.println(canonicalizeKey(rdep));
        }
      } else {
        out.println("  (rdeps not stored)");
      }
      out.println();
    }
  }

  private String canonicalizeKey(SkyKey key) {
    return String.format(
        "%s:%s\n", key.functionName(), key.argument().toString().replace('\n', '_'));
  }
}
