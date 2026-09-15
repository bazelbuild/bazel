// Copyright 2024 The Bazel Authors. All rights reserved.
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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.analysis.ConfiguredRuleClassProvider;
import com.google.devtools.build.lib.collect.ConcurrentIdentitySet;
import com.google.devtools.build.lib.util.MemoryAccountant;
import com.google.devtools.build.lib.util.MemoryAccountant.Stats;
import com.google.devtools.build.lib.util.ObjectGraphTraverser;
import com.google.devtools.build.lib.util.ObjectGraphTraverser.FieldCache;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.NodeEntry;
import com.google.devtools.build.skyframe.QueryableGraph.Reason;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.io.PrintStream;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.function.BiConsumer;

/** An utility to dump the memory use of the objects in Skyframe in various ways. */
public class SkyframeMemoryDumper {

  /** How to display Skyframe memory use. */
  public enum DisplayMode {
    /** Just a summary line */
    SUMMARY,
    /** Object count by class */
    COUNT,
    /** Bytes by class */
    BYTES,
  }

  /** An exception that signals that dumping Skyframe memory did not work out. */
  public static class DumpFailedException extends Exception {
    public DumpFailedException(String message, Throwable cause) {
      super(message, cause);
    }
  }

  // Fields affecting how the results are displayed
  private final DisplayMode displayMode;
  private final String needle;

  // Fields affecting how the data is collected
  private final boolean reportTransient;
  private final ConfiguredRuleClassProvider ruleClassProvider;
  private final FieldCache fieldCache;
  private final ImmutableList<MemoryAccountant.Measurer> measurers;

  // Data that is being dumped
  private final InMemoryGraph graph;

  public SkyframeMemoryDumper(
      DisplayMode displayMode,
      String needle,
      ConfiguredRuleClassProvider ruleClassProvider,
      InMemoryGraph graph,
      boolean reportTransient,
      boolean reportConfiguration,
      boolean reportPrecomputed,
      boolean reportWorkspaceStatus) {
    this.displayMode = displayMode;
    this.needle = needle;
    BuildObjectTraverser buildObjectTraverser =
        new BuildObjectTraverser(reportConfiguration, reportPrecomputed, reportWorkspaceStatus);
    CollectionObjectTraverser collectionObjectTraverser = new CollectionObjectTraverser();
    this.graph = graph;
    this.ruleClassProvider = ruleClassProvider;
    this.fieldCache =
        new FieldCache(ImmutableList.of(buildObjectTraverser, collectionObjectTraverser));
    this.measurers = ImmutableList.of(collectionObjectTraverser);
    this.reportTransient = reportTransient;
  }

  private static String jsonQuote(String s) {
    try {
      StringWriter writer = new StringWriter();
      JsonWriter json = new JsonWriter(writer);
      json.value(s);
      json.flush();
      return writer.toString();
    } catch (IOException e) {
      // StringWriter does no I/O
      throw new IllegalStateException(e);
    }
  }

  public static void printByClass(String prefix, Map<String, Long> memory, PrintStream out) {
    out.print("{");

    ImmutableList<Entry<String, Long>> sorted =
        memory.entrySet().stream()
            .sorted(Comparator.comparing(Entry<String, Long>::getValue).reversed())
            .collect(ImmutableList.toImmutableList());

    boolean first = true;
    for (Entry<String, Long> entry : sorted) {
      out.printf(
          "%s\n%s  %s: %d", first ? "" : ",", prefix, jsonQuote(entry.getKey()), entry.getValue());
      first = false;
    }

    out.printf("\n%s}", prefix);
  }

  private void addBuiltins(ConcurrentIdentitySet set) {
    ObjectGraphTraverser traverser =
        new ObjectGraphTraverser(
            this.fieldCache,
            /* countInternedObjects= */ false,
            /* reportTransientFields= */ true,
            set,
            /* collectContext= */ false,
            ObjectGraphTraverser.NOOP_OBJECT_RECEIVER,
            /* instanceId= */ null);
    traverser.traverse(ruleClassProvider);
  }

  public Stats dumpShallow(NodeEntry nodeEntry) throws InterruptedException {
    ConcurrentIdentitySet seenObjects = new ConcurrentIdentitySet(1);
    addBuiltins(seenObjects);

    // Mark all objects accessible from direct dependencies. This will mutate seen, but that's OK.
    for (SkyKey directDepKey : nodeEntry.getDirectDeps()) {
      NodeEntry directDepEntry = graph.get(null, Reason.OTHER, directDepKey);
      ObjectGraphTraverser depTraverser =
          new ObjectGraphTraverser(
              fieldCache,
              false,
              reportTransient,
              seenObjects,
              false,
              ObjectGraphTraverser.NOOP_OBJECT_RECEIVER,
              null);
      depTraverser.traverse(directDepEntry.getValue());
    }

    // Now traverse the objects reachable from the given SkyValue. Objects reachable from direct
    // dependencies are in "seen" and thus will not be counted.
    return dumpReachable(nodeEntry, seenObjects);
  }

  public Stats dumpReachable(NodeEntry nodeEntry, ConcurrentIdentitySet seenObjects)
      throws InterruptedException {
    MemoryAccountant memoryAccountant =
        new MemoryAccountant(measurers, displayMode != DisplayMode.SUMMARY);
    ObjectGraphTraverser traverser =
        new ObjectGraphTraverser(
            fieldCache, true, reportTransient, seenObjects, true, memoryAccountant, null, needle);
    traverser.traverse(nodeEntry.getValue());
    return memoryAccountant.getStats();
  }

  public Stats dumpReachable(NodeEntry nodeEntry) throws InterruptedException {
    ConcurrentIdentitySet seenObjects = new ConcurrentIdentitySet(1);
    addBuiltins(seenObjects);
    return dumpReachable(nodeEntry, seenObjects);
  }

  private ListenableFuture<Void> processTransitive(
      BiConsumer<SkyKey, SkyValue> processor,
      SkyKey skyKey,
      Executor executor,
      Map<SkyKey, ListenableFuture<Void>> futureMap) {

    // This is awkward, but preferable to plumbing this through scheduleDeps and processDeps
    SkyValue[] value = new SkyValue[1];

    // First get the SkyValue and the direct deps from the Skyframe graph. This happens in a future
    // so that processTransitive() (which is called from computeIfAbsent()) doesn't throw a
    // checked exception.
    ListenableFuture<Iterable<SkyKey>> fetchNodeData =
        Futures.submit(
            () -> {
              NodeEntry entry = graph.get(null, Reason.OTHER, skyKey);
              value[0] = entry.getValue();
              return entry.getDirectDeps();
            },
            executor);

    // This returns a list of futures representing processing the direct deps of this node
    ListenableFuture<ImmutableList<ListenableFuture<Void>>> scheduleDeps =
        Futures.transform(
            fetchNodeData,
            directDeps -> {
              List<ListenableFuture<Void>> depFutures = new ArrayList<>();
              for (SkyKey dep : directDeps) {
                // If the processing of this dependency has not been scheduled, do so
                depFutures.add(
                    futureMap.computeIfAbsent(
                        dep, k -> processTransitive(processor, dep, executor, futureMap)));
              }
              return ImmutableList.copyOf(depFutures);
            },
            executor);

    // This is a future that gets completed when the direct deps have all been processed...
    ListenableFuture<List<Void>> processDeps =
        Futures.transformAsync(scheduleDeps, Futures::allAsList, executor);

    // ...and when that's the case, we can proceed with processing this node in turn.
    return Futures.transform(
        processDeps,
        unused -> {
          processor.accept(skyKey, value[0]);
          return null;
        },
        executor);
  }

  private static ExecutorService createExecutor() {
    return Executors.newFixedThreadPool(
        Runtime.getRuntime().availableProcessors(),
        new ThreadFactoryBuilder().setNameFormat("dump-ram-%d").build());
  }

  public Stats dumpTransitive(SkyKey skyKey) throws InterruptedException {
    ConcurrentIdentitySet seenObjects = new ConcurrentIdentitySet(1);
    addBuiltins(seenObjects);

    MemoryAccountant memoryAccountant =
        new MemoryAccountant(measurers, displayMode != DisplayMode.SUMMARY);
    BiConsumer<SkyKey, SkyValue> processor =
        (unused, skyValue) -> {
          ObjectGraphTraverser traverser =
              new ObjectGraphTraverser(
                  fieldCache,
                  false,
                  reportTransient,
                  seenObjects,
                  true,
                  memoryAccountant,
                  null,
                  needle);
          traverser.traverse(skyValue);
        };

    try (ExecutorService executor = createExecutor()) {
      ListenableFuture<Void> work =
          processTransitive(processor, skyKey, executor, new ConcurrentHashMap<>());
      work.get();
    } catch (ExecutionException e) {
      throw new IllegalStateException(e);
    }

    return memoryAccountant.getStats();
  }

  public void dumpFull(PrintStream out) throws InterruptedException, DumpFailedException {
    // Profiling shows that the average object count for a Skyframe node is around 30-40. Let's
    // go with 48 to avoid a potentially costly resize.
    ConcurrentIdentitySet seenObjects =
        new ConcurrentIdentitySet(graph.getAllNodeEntries().size() * 48);
    addBuiltins(seenObjects);

    ImmutableList<SkyKey> roots =
        graph.getAllNodeEntries().parallelStream()
            .filter(e -> e.isDone() && Iterables.isEmpty(e.getReverseDepsForDoneEntry()))
            .map(InMemoryNodeEntry::getKey)
            .collect(ImmutableList.toImmutableList());

    ConcurrentHashMap<SkyKey, Stats> nodeStats = new ConcurrentHashMap<>();

    BiConsumer<SkyKey, SkyValue> processor =
        (skyKey, skyValue) -> {
          MemoryAccountant memoryAccountant =
              new MemoryAccountant(measurers, displayMode != DisplayMode.SUMMARY);
          ObjectGraphTraverser traverser =
              new ObjectGraphTraverser(
                  fieldCache,
                  true,
                  reportTransient,
                  seenObjects,
                  true,
                  memoryAccountant,
                  skyKey,
                  needle);
          traverser.traverse(skyValue);
          Stats stats = memoryAccountant.getStats();
          nodeStats.put(skyKey, stats);
        };

    try (ExecutorService executor = createExecutor()) {
      ConcurrentHashMap<SkyKey, ListenableFuture<Void>> futureMap =
          new ConcurrentHashMap<>(128, 0.75f, Runtime.getRuntime().availableProcessors());
      ImmutableList<ListenableFuture<Void>> rootFutures =
          roots.stream()
              .map(l -> processTransitive(processor, l, executor, futureMap))
              .collect(ImmutableList.toImmutableList());

      ListenableFuture<List<Void>> completion = Futures.allAsList(rootFutures);
      completion.get();
    } catch (ExecutionException e) {
      throw new DumpFailedException("Error during traversal: " + e.getMessage(), e);
    }

    var sortedStats =
        nodeStats.entrySet().stream()
            .parallel()
            .map(e -> Pair.of(e.getKey().getCanonicalName(), e.getValue()))
            .sorted(Comparator.comparing(Pair::getFirst))
            .collect(ImmutableList.toImmutableList());

    out.print("{");
    boolean first = true;
    for (Pair<String, Stats> p : sortedStats) {
      out.printf("%s\n  %s: ", first ? "" : ",", jsonQuote(p.getFirst()));
      Stats v = p.getSecond();
      first = false;
      switch (displayMode) {
        case SUMMARY ->
            out.printf("{ \"objects\": %d, \"bytes\": %d }", v.getObjectCount(), v.getMemoryUse());
        case COUNT -> printByClass("  ", v.getObjectCountByClass(), out);
        case BYTES -> printByClass("  ", v.getMemoryByClass(), out);
      }
    }
    out.println("\n}");
  }
}
