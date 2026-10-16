// Copyright 2025 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import static com.google.common.collect.ImmutableList.toImmutableList;
import static com.google.common.collect.ImmutableSet.toImmutableSet;
import static com.google.common.truth.Truth.assertThat;
import static com.google.common.truth.Truth.assertWithMessage;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;
import static org.junit.Assert.assertThrows;

import com.google.common.base.Function;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.analysis.ConfiguredTarget;
import com.google.devtools.build.lib.buildtool.util.BuildIntegrationTestCase;
import com.google.devtools.build.lib.cmdline.Label;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutorOwner;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodesWithSource;
import com.google.devtools.build.lib.skyframe.ConfiguredTargetKey;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RejectedExecutionException;
import java.util.function.Predicate;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.JUnit4;

@RunWith(JUnit4.class)
public final class FileOpNodeMemoizingLookupTest extends BuildIntegrationTestCase {
  // TODO: b/364831651 - consider adding test cases covering other scenarios, like symlinks.

  private static final int CONCURRENCY = 4;

  private ForkJoinPool forkJoinPool;

  @Before
  public void createForkJoinPool() {
    forkJoinPool = new ForkJoinPool(CONCURRENCY);
  }

  @After
  public void shutdownForkJoinPool() {
    forkJoinPool.shutdownNow();
  }

  @Test
  public void fileOpNodes_areConsistent() throws Exception {
    // This test case contains a glob to exercise DirectoryListingKey.
    write("hello/x.txt", "x");
    write(
        "hello/BUILD",
        """
        genrule(
            name = "target",
            srcs = glob(["*.txt"]),
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//hello:target");

    InMemoryGraph graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();

    var pool = new SafeExecutorOwner(new ForkJoinPool(CONCURRENCY));

    var fileOpDataMap =
        new FileOpNodeMemoizingLookup(
            pool,
            graph,
            ImmutableSet.of(),
            /* shouldDiscardMemory= */ false,
            /* referencedPackages= */ null);

    var actionLookups = new ArrayList<ActionLookupKey>();
    var actions = new ArrayList<ActionLookupData>();

    for (SkyKey key : graph.getDoneValues().keySet()) {
      if (key instanceof ActionLookupKey lookupKey) {
        actionLookups.add(lookupKey);
      }
      if (key instanceof ActionLookupData lookupData) {
        actions.add(lookupData);
      }
    }

    var futures = new ConcurrentLinkedQueue<ListenableFuture<Void>>();
    var allAdded = new CountDownLatch(actionLookups.size() + actions.size());

    for (ActionLookupKey lookupKey : actionLookups) {
      pool.execute(
          new RejectionHandlingRunnable() {
            @Override
            public void run() {
              futures.add(verifyFileOpNodeForActionLookupKey(graph, fileOpDataMap, lookupKey));
              allAdded.countDown();
            }

            @Override
            public void handleRejection(Throwable t) {
              allAdded.countDown();
            }
          });
    }
    for (ActionLookupData lookupData : actions) {
      pool.execute(
          new RejectionHandlingRunnable() {
            @Override
            public void run() {
              futures.add(verifyFileOpNodeForActionLookupData(graph, fileOpDataMap, lookupData));
              allAdded.countDown();
            }

            @Override
            public void handleRejection(Throwable t) {
              allAdded.countDown();
            }
          });
    }

    allAdded.await();
    // Should not raise any exceptions.
    var unused = Futures.whenAllSucceed(futures).call(() -> null, directExecutor()).get();
  }

  private static ListenableFuture<Void> verifyFileOpNodeForActionLookupKey(
      InMemoryGraph graph, FileOpNodeMemoizingLookup fileOpDataMap, ActionLookupKey lookupKey) {
    // For action lookup values, verifies that the file dependencies are an exact match for the ones
    // in the transitive closure.
    Function<FileOpNodeOrEmpty, Void> verify =
        node -> {
          var nodes = new HashSet<FileOpNode>();
          var sources = new HashSet<FileKey>();
          flattenNodeOrEmpty(node, nodes, sources, new HashSet<>());
          assertWithMessage("for key=%s", lookupKey)
              .that(nodes)
              .isEqualTo(collectTransitiveFileOpNodes(graph, lookupKey));
          return null;
        };
    FileOpNodeOrFuture nodeOrFuture = fileOpDataMap.computeNode(lookupKey);
    switch (nodeOrFuture) {
      case FileOpNodeOrEmpty nodeOrEmpty -> {
        var unusedNull = verify.apply(nodeOrEmpty);
        return immediateVoidFuture();
      }
      case FutureFileOpNode future -> {
        return Futures.transform(future, verify, directExecutor());
      }
    }
  }

  private static ListenableFuture<Void> verifyFileOpNodeForActionLookupData(
      InMemoryGraph graph, FileOpNodeMemoizingLookup fileOpDataMap, ActionLookupData lookupData) {
    // For actions, verifies that the union of the files and sources of the file op data of the
    // action's owner is a superset of the file dependencies of the action. There's a small
    // overapproximation here.
    Function<FileOpNodeOrEmpty, Void> verify =
        node -> {
          var nodes = new HashSet<FileOpNode>();
          var sources = new HashSet<FileKey>();
          flattenNodeOrEmpty(node, nodes, sources, new HashSet<>());
          ImmutableSet<FileOpNode> realFileDeps = collectTransitiveFileOpNodes(graph, lookupData);

          var assertBuilder = assertWithMessage("for key=%s", lookupData);
          assertBuilder.that(nodes).containsNoneIn(sources); // Sources are distinct from nodes.
          // All sources are contained in the real file deps.
          assertBuilder.that(realFileDeps).containsAtLeastElementsIn(sources);

          nodes.addAll(sources);
          // Sources may be an overapproximation by design. In this particular case, it happens to
          // be an exact match, but that could conceivably change with code changes.
          assertBuilder.that(nodes).containsAtLeastElementsIn(realFileDeps);
          return null;
        };
    // Note, that this looks up the incrementality data for the action by its ActionLookupKey.
    FileOpNodeOrFuture nodeOrFuture = fileOpDataMap.computeNode(lookupData.getActionLookupKey());
    switch (nodeOrFuture) {
      case FileOpNodeOrEmpty nodeOrEmpty -> {
        var unusedNull = verify.apply(nodeOrEmpty);
        return immediateVoidFuture();
      }
      case FutureFileOpNode future -> {
        return Futures.transform(future, verify, directExecutor());
      }
    }
  }

  /**
   * Flattens the given node or empty node into the given sets of nodes and sources.
   *
   * <p>The given sets are modified in place.
   */
  private static void flattenNodeOrEmpty(
      FileOpNodeOrEmpty maybeNode,
      Set<FileOpNode> nodes,
      Set<FileKey> sources,
      Set<FileOpNode> visited) {
    switch (maybeNode) {
      case EMPTY_FILE_OP_NODE -> {
        return;
      }
      case FileOpNode node -> {
        flattenNode(node, nodes, sources, visited);
        return;
      }
    }
  }

  private static void flattenNode(
      FileOpNode node, Set<FileOpNode> nodes, Set<FileKey> sources, Set<FileOpNode> visited) {
    if (!visited.add(node)) {
      return;
    }
    switch (node) {
      case FileKey file -> nodes.add(file);
      case DirectoryListingKey directory -> nodes.add(directory);
      case RemoteFileOpNode remote -> nodes.add(remote);
      case NestedFileOpNodes nested -> {
        for (int i = 0; i < nested.analysisDependenciesCount(); i++) {
          flattenNode(nested.getAnalysisDependency(i), nodes, sources, visited);
        }
      }
      case NestedFileOpNodesWithSource withSources -> {
        for (int i = 0; i < withSources.analysisDependenciesCount(); i++) {
          flattenNode(withSources.getAnalysisDependency(i), nodes, sources, visited);
        }
        sources.add(withSources.source());
      }
    }
  }

  private static ImmutableSet<FileOpNode> collectTransitiveFileOpNodes(
      InMemoryGraph graph, SkyKey key) {
    var visited = new HashSet<SkyKey>();
    var nodes = new HashSet<FileOpNode>();
    collectTransitiveFileOpNodes(graph, key, visited, nodes);
    return ImmutableSet.copyOf(nodes);
  }

  private static void collectTransitiveFileOpNodes(
      InMemoryGraph graph, SkyKey key, Set<SkyKey> visited, Set<FileOpNode> nodes) {
    if (!visited.add(key)) {
      return;
    }
    if (key instanceof FileOpNode fileOp) {
      // The FileOpNodeMemoizingLookup doesn't recurse beyond FileKey or DirectoryListingKeys. The
      // inner details
      // of those entries are handled by FileDependencySerializer.
      nodes.add(fileOp);
      return;
    }
    for (SkyKey dep : graph.getIfPresent(key).getDirectDeps()) {
      collectTransitiveFileOpNodes(graph, dep, visited, nodes);
    }
  }

  @Test
  public void fileOpNodes_unselectedAreDiscarded() throws Exception {
    write("discard/x.txt", "x");
    write("discard/y.txt", "y");
    write(
        "discard/BUILD",
        """
        genrule(
            name = "target",
            srcs = ["x.txt", "y.txt"],
            outs = ["out"],
            cmd = "cat $(SRCS) > $@",
        )
        """);

    buildTarget("//discard:target");

    InMemoryGraph graph = getSkyframeExecutor().getEvaluator().getInMemoryGraph();

    var pool = new SafeExecutorOwner(new ForkJoinPool(CONCURRENCY));

    ImmutableList<ActionLookupKey> actionLookups =
        graph.getDoneValues().keySet().stream()
            .filter(ActionLookupKey.class::isInstance)
            .map(ActionLookupKey.class::cast)
            .collect(toImmutableList());

    assertThat(actionLookups.size()).isAtLeast(2); // At least one selected and one unselected.
    ActionLookupKey selectedKey = actionLookups.get(0);
    ImmutableSet<SkyKey> selectedKeys = ImmutableSet.of(selectedKey);

    ImmutableList<ActionLookupKey> unselectedKeys =
        actionLookups.stream()
            .filter(key -> !selectedKeys.contains(key))
            .collect(toImmutableList());
    assertThat(unselectedKeys).isNotEmpty();

    // Referenced PackageIdentifier keys should NOT be discarded.
    ImmutableList<PackageIdentifier> referencedPackageKeys =
        graph.getDoneValues().keySet().stream()
            .filter(PackageIdentifier.class::isInstance)
            .map(PackageIdentifier.class::cast)
            .collect(toImmutableList());
    assertThat(referencedPackageKeys).isNotEmpty();

    var fileOpDataMap =
        new FileOpNodeMemoizingLookup(
            pool,
            graph,
            selectedKeys,
            /* shouldDiscardMemory= */ true,
            /* referencedPackages= */ ImmutableSet.copyOf(referencedPackageKeys));

    var futures = new ConcurrentLinkedQueue<ListenableFuture<Void>>();
    for (ActionLookupKey key : actionLookups) {
      switch (fileOpDataMap.computeNode(key)) {
        case FileOpNodeOrEmpty nodeOrEmpty -> futures.add(Futures.immediateVoidFuture());
        case FutureFileOpNode future ->
            futures.add(Futures.transform(future, unused -> null, directExecutor()));
      }
    }
    var unused = Futures.whenAllSucceed(futures).call(() -> null, directExecutor()).get();

    // The selected key should still be present in the graph.
    assertThat(graph.getIfPresent(selectedKey)).isNotNull();

    // Unselected keys should be discarded (i.e. graph.getIfPresent returns null).
    for (ActionLookupKey key : unselectedKeys) {
      assertWithMessage("unselected key %s should be discarded", key)
          .that(graph.getIfPresent(key))
          .isNull();
    }

    for (PackageIdentifier key : referencedPackageKeys) {
      assertWithMessage("referenced PackageIdentifier key %s should NOT be discarded", key)
          .that(graph.getIfPresent(key))
          .isNotNull();
    }
  }

  @Test
  public void materializeNodeGraph_missingDependency_returnsOneFailurePerFailedNode()
      throws Exception {
    write("pkg/a.txt", "a");
    write("pkg/b.txt", "b");
    write(
        "pkg/BUILD",
        genrule("ok", "['a.txt']") + genrule("bad1", "['b.txt']") + genrule("bad2", "['b.txt']"));
    buildTarget("//pkg:ok", "//pkg:bad1", "//pkg:bad2");
    ConfiguredTargetKey bad1Key = ctKey("//pkg:bad1");
    ImmutableSet<SkyKey> selection =
        ImmutableSet.of(ctKey("//pkg:ok"), bad1Key, ctKey("//pkg:bad2"));
    SkyKey removed = removeDirectDependency(bad1Key, "b.txt");
    var lookup = newLookup(selection);

    // Both bad roots fail on the same missing node. The failure is reported once, deduplicated by
    // identity, and materialization does not throw.
    assertSingleMissingEntry(lookup.materializeNodeGraph(selection), removed);
  }

  @Test
  public void materializeNodeGraph_emptySelection_completesSuccessfully() throws Exception {
    assertThat(newLookup(ImmutableSet.of()).materializeNodeGraph(ImmutableSet.of())).isEmpty();
  }

  @Test
  public void materializeNodeGraph_interrupted_throwsInterruptedException() throws Exception {
    var lookup = newLookup(ImmutableSet.of());

    Thread.currentThread().interrupt();
    try {
      assertThrows(
          InterruptedException.class, () -> lookup.materializeNodeGraph(ImmutableSet.of()));
    } finally {
      // Clear interrupted status
      Thread.interrupted();
    }
  }

  @Test
  public void materializeNodeGraph_unexpectedExceptionInBatch_returnsException() throws Exception {
    write("pkg/a.txt", "a");
    write("pkg/BUILD", genrule("ok", "['a.txt']"));
    buildTarget("//pkg:ok");
    // FileKey is in the graph, so isSkipped returns false, but getDependencyKey does not support
    // it.
    SkyKey unhandledKey = doneKeys(key -> key instanceof FileKey).iterator().next();
    var lookup = newLookup(ImmutableSet.of(unhandledKey));

    ImmutableList<Throwable> failures = lookup.materializeNodeGraph(ImmutableSet.of(unhandledKey));

    assertThat(failures).hasSize(1);
    assertThat(failures.get(0)).isInstanceOf(IllegalStateException.class);
    assertThat(failures.get(0)).hasMessageThat().contains("unexpected key");
  }

  @Test
  public void materializeNodeGraph_batchFailure_stillCollectsNodeFailures() throws Exception {
    write("pkg/a.txt", "a");
    write("pkg/b.txt", "b");
    write("pkg/BUILD", genrule("ok", "['a.txt']") + genrule("bad", "['b.txt']"));
    buildTarget("//pkg:ok", "//pkg:bad");
    ConfiguredTargetKey badKey = ctKey("//pkg:bad");
    SkyKey removed = removeDirectDependency(badKey, "b.txt");
    SkyKey unhandledKey = doneKeys(key -> key instanceof FileKey).iterator().next();
    ImmutableSet<SkyKey> selection = ImmutableSet.of(unhandledKey, ctKey("//pkg:ok"), badKey);
    var lookup = newLookup(selection);

    // A throwing batch does not end materialization early: the node failure is still reported.
    ImmutableList<Throwable> failures = lookup.materializeNodeGraph(selection);

    assertThat(failures.stream().map(Object::getClass))
        .containsExactly(IllegalStateException.class, MissingSkyframeEntryException.class);
    assertThat(
            failures.stream()
                .filter(MissingSkyframeEntryException.class::isInstance)
                .map(e -> ((MissingSkyframeEntryException) e).key()))
        .containsExactly(removed);
  }

  @Test
  public void materializeNodeGraph_rejectedExecution_returnsRejection() throws Exception {
    forkJoinPool.shutdown();
    ActionLookupKey absentKey = unbuiltKey("//pkg:absent");
    var lookup = newLookup(ImmutableSet.of(absentKey));

    ImmutableList<Throwable> failures = lookup.materializeNodeGraph(ImmutableSet.of(absentKey));

    assertThat(failures).hasSize(1);
    assertThat(failures.get(0)).isInstanceOf(RejectedExecutionException.class);
  }

  @Test
  public void materializeNodeGraph_cancelledNode_returnsCancellationException() throws Exception {
    write("pkg/a.txt", "a");
    write("pkg/b.txt", "b");
    write("pkg/BUILD", genrule("ok", "['a.txt']") + genrule("cancelled", "['b.txt']"));
    buildTarget("//pkg:ok", "//pkg:cancelled");
    ConfiguredTargetKey cancelledKey = ctKey("//pkg:cancelled");
    ImmutableSet<SkyKey> selection = ImmutableSet.of(ctKey("//pkg:ok"), cancelledKey);
    var executor = new DeferredCallbackExecutor(new SafeExecutorOwner(forkJoinPool));
    var lookup = newLookup(executor, selection);

    // Pre-populate cancelledKey as an in-flight future with an unbuilt child dependency, then
    // cancel it.
    FileOpNodeOrFuture inFlight =
        lookup.computeNode(
            cancelledKey, new SkyValue() {}, ImmutableList.of(unbuiltKey("//pkg:child")));
    assertThat(inFlight).isInstanceOf(FutureFileOpNode.class);
    executor.failOnlyPendingCallback(new CancellationException("injected node cancellation"));
    assertThat(((FutureFileOpNode) inFlight).isCancelled()).isTrue();

    // materializeNodeGraph completes normally, reporting the cancelled future.
    ImmutableList<Throwable> failures = lookup.materializeNodeGraph(selection);

    assertThat(failures).hasSize(1);
    assertThat(failures.get(0)).isInstanceOf(CancellationException.class);
  }

  @Test
  public void computeNode_afterDirectDepsCleared_failsUnlessMaterialized() throws Exception {
    write("pkg/a.txt", "a");
    write("pkg/b.txt", "b");
    write(
        "pkg/BUILD", genrule("materialized", "['a.txt']") + genrule("unmaterialized", "['b.txt']"));
    buildTarget("//pkg:materialized", "//pkg:unmaterialized");
    ConfiguredTargetKey materializedKey = ctKey("//pkg:materialized");
    ConfiguredTargetKey unmaterializedKey = ctKey("//pkg:unmaterialized");
    var lookup = newLookup(ImmutableSet.of(materializedKey));
    assertThat(lookup.materializeNodeGraph(ImmutableSet.of(materializedKey))).isEmpty();

    lookup.markDirectDepsCleared();

    // A memoized node is still served.
    assertThat(lookup.computeNode(materializedKey)).isInstanceOf(FileOpNodeOrEmpty.class);
    // Any other node would read empty deps from the graph, so it fails instead.
    var future = (FutureFileOpNode) lookup.computeNode(unmaterializedKey);
    var thrown = assertThrows(ExecutionException.class, future::get);
    assertThat(thrown).hasCauseThat().isInstanceOf(IllegalStateException.class);
    assertThat(thrown)
        .hasCauseThat()
        .hasMessageThat()
        .contains("was not materialized before Skyframe direct deps were cleared");
  }

  @Test
  public void populateFutureFileOpNode_cancelledCollector_cancelsOwnedFuture() {
    SafeExecutor cancellingCallbackExecutor =
        new SafeExecutor() {
          @Override
          public void execute(RejectionHandlingRunnable task) {
            task.run();
          }

          @Override
          public <V> void addCallback(
              ListenableFuture<V> future, FutureCallback<? super V> callback) {
            callback.onFailure(new CancellationException("injected dep cancellation"));
          }

          @Override
          public Executor getInternalUnsafeExecutor() {
            return Runnable::run;
          }
        };
    ActionLookupKey parentKey = unbuiltKey("//pkg:parent");
    var lookup = newLookup(cancellingCallbackExecutor, ImmutableSet.of(parentKey));

    FileOpNodeOrFuture result =
        lookup.computeNode(
            parentKey, new SkyValue() {}, ImmutableList.of(unbuiltKey("//pkg:child")));

    assertThat(result).isInstanceOf(FutureFileOpNode.class);
    var future = (FutureFileOpNode) result;
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  @Test
  public void populateFutureFileOpNode_asynchronouslyCancelledCollector_cancelsOwnedFuture() {
    var executor = new DeferredCallbackExecutor(SafeExecutor.safeDirectExecutor());
    ActionLookupKey parentKey = unbuiltKey("//pkg:parent");
    var lookup = newLookup(executor, ImmutableSet.of(parentKey));

    FileOpNodeOrFuture result =
        lookup.computeNode(
            parentKey, new SkyValue() {}, ImmutableList.of(unbuiltKey("//pkg:child")));

    assertThat(result).isInstanceOf(FutureFileOpNode.class);
    var future = (FutureFileOpNode) result;
    assertThat(future.isDone()).isFalse();

    executor.failOnlyPendingCallback(new CancellationException("injected async cancellation"));
    assertThat(future.isCancelled()).isTrue();
    assertThrows(CancellationException.class, future::get);
  }

  /**
   * A {@link SafeExecutor} that holds on to the callbacks registered with it, so that a test can
   * fail them at a chosen point.
   */
  private static final class DeferredCallbackExecutor implements SafeExecutor {
    private final SafeExecutor taskExecutor;
    private final List<FutureCallback<?>> pendingCallbacks = new ArrayList<>();

    private DeferredCallbackExecutor(SafeExecutor taskExecutor) {
      this.taskExecutor = taskExecutor;
    }

    @Override
    public void execute(RejectionHandlingRunnable task) {
      taskExecutor.execute(task);
    }

    @Override
    public synchronized <V> void addCallback(
        ListenableFuture<V> future, FutureCallback<? super V> callback) {
      pendingCallbacks.add(callback);
    }

    @Override
    public Executor getInternalUnsafeExecutor() {
      return Runnable::run;
    }

    /** Fails the one callback registered so far. */
    synchronized void failOnlyPendingCallback(Throwable t) {
      assertThat(pendingCallbacks).hasSize(1);
      pendingCallbacks.get(0).onFailure(t);
    }
  }

  /** A genrule concatenating {@code srcs}, a Starlark list or glob expression. */
  private static String genrule(String name, String srcs) {
    return String.format(
        """
        genrule(
            name = "%1$s",
            srcs = %2$s,
            outs = ["%1$s.out"],
            cmd = "cat $(SRCS) > $@",
        )
        """,
        name, srcs);
  }

  private InMemoryGraph graph() {
    return getSkyframeExecutor().getEvaluator().getInMemoryGraph();
  }

  private ImmutableSet<SkyKey> doneKeys(Predicate<SkyKey> filter) {
    return graph().getDoneValues().keySet().stream().filter(filter).collect(toImmutableSet());
  }

  private FileOpNodeMemoizingLookup newLookup(
      SafeExecutor executor, ImmutableSet<SkyKey> selection) {
    return new FileOpNodeMemoizingLookup(
        executor,
        graph(),
        selection,
        /* shouldDiscardMemory= */ false,
        /* referencedPackages= */ null);
  }

  private FileOpNodeMemoizingLookup newLookup(ImmutableSet<SkyKey> selection) {
    return newLookup(new SafeExecutorOwner(forkJoinPool), selection);
  }

  private ConfiguredTargetKey ctKey(String label) throws Exception {
    ConfiguredTarget target = getConfiguredTarget(label);
    assertThat(target).isNotNull();
    return ConfiguredTargetKey.fromConfiguredTarget(target);
  }

  /** A key for a target that was never built, and so is absent from the graph. */
  private static ActionLookupKey unbuiltKey(String label) {
    return ConfiguredTargetKey.builder().setLabel(Label.parseCanonicalUnchecked(label)).build();
  }

  /**
   * Removes from the graph the first direct dependency of {@code parent} whose key contains {@code
   * fragment}, and returns its key.
   */
  private SkyKey removeDirectDependency(SkyKey parent, String fragment) throws Exception {
    InMemoryGraph graph = graph();
    for (SkyKey dep : graph.getIfPresent(parent).getDirectDeps()) {
      SkyKey dependencyKey = dep instanceof ActionLookupData ald ? ald.getActionLookupKey() : dep;
      if (dependencyKey.toString().contains(fragment)
          && graph.getIfPresent(dependencyKey) != null) {
        graph.remove(dependencyKey);
        return dependencyKey;
      }
    }
    throw new AssertionError("no direct dependency of " + parent + " matches " + fragment);
  }

  private static void assertSingleMissingEntry(ImmutableList<Throwable> failures, SkyKey key) {
    assertThat(failures).hasSize(1);
    assertThat(failures.get(0)).isInstanceOf(MissingSkyframeEntryException.class);
    assertThat(((MissingSkyframeEntryException) failures.get(0)).key()).isEqualTo(key);
  }
}
