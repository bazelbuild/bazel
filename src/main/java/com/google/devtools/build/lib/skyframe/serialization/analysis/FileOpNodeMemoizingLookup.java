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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor.safeDirectExecutor;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;
import static java.lang.Math.max;
import static java.lang.Math.min;

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.AccumulatingQuiescingFuture;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.concurrent.safeexecutor.RejectionHandlingRunnable;
import com.google.devtools.build.lib.concurrent.safeexecutor.SafeExecutor;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode;
import com.google.devtools.build.lib.skyframe.NonRuleConfiguredTargetValue;
import com.google.devtools.build.lib.skyframe.config.BaselineOptionsFunction;
import com.google.devtools.build.lib.skyframe.serialization.DeserializedSkyValue;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import com.google.errorprone.annotations.CheckReturnValue;
import com.google.errorprone.annotations.DoNotCall;
import com.google.protobuf.ByteString;
import java.util.Set;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import javax.annotation.Nullable;

/**
 * Computes a mapping from {@link ActionLookupKey}s to {@link FileOpNodeOrFuture}s, representing the
 * complete set of file system operation dependencies required to evaluate each key.
 *
 * <p>This class tracks file dependencies for a particular build. It uses the file and source
 * partitioning in {@link AbstractNestedFileOpNodes} to provide a view of file dependencies for
 * configured targets and actions. For configured targets, only the analysis dependencies (BUILD,
 * .bzl files) are relevant. For actions, the source (.h, .cpp, .java) files must also be
 * considered.
 *
 * <p><b>Approximation for Efficiency:</b> To avoid the excessive overhead of storing precise file
 * dependencies per action, an over-approximation is used. This may lead to occasional spurious
 * cache misses but guarantees no false cache hits. The approximation includes all source
 * dependencies declared by the configured target that were visited during the build.
 *
 * <p>Not all actions of a configured target are executed, and include scanning may eliminate
 * dependencies, so the actual set of source files visited by a build may be a subset of the
 * declared ones. This will never skip an actual action file dependency of the build. While this is
 * correct, it's possible that different builds at the same version will have slightly different
 * representations of the sets of sources.
 *
 * <p><b>Why Approximation?</b> <br>
 * Storing the exact file dependencies for each action individually would be too expensive. It would
 * negate the benefits of the compact nested representation used for configured target dependencies.
 * The chosen approximation balances accuracy with performance.
 *
 * <p><b>Different Sources in Multiple Builds</b> <br>
 * Suppose there are multiple builds that share configured targets, but request different actions
 * from those configured targets. The configured target data is deterministic and shared, but the
 * invalidation information for source files could differ. When invalidating the configured target,
 * the source files are ignored, so even if a second build overwrites the configured target of the
 * first, invalidation of the configured target still works exactly the same way. For actions,
 * overwriting of the configured target doesn't affect correctness either because each action
 * directly references the invalidation data created by its respective build.
 */
public final class FileOpNodeMemoizingLookup {

  /**
   * An {@link AbstractValueOrFutureMap} that allows passing in data that it forwards to the
   * function computing the value from the key.
   */
  private final class FileOpNodeMap
      extends AbstractValueOrFutureMap<
          SkyKey, FileOpNodeOrFuture, FileOpNodeOrEmpty, FutureFileOpNode> {

    private FileOpNodeMap() {
      super(new ConcurrentHashMap<>(), FutureFileOpNode::new, FutureFileOpNode.class);
    }

    private FileOpNodeOrFuture getValueOrFuture(SkyKey key) {
      return getValueOrFuture(key, null, null);
    }

    private FileOpNodeOrFuture getValueOrFuture(
        SkyKey key, @Nullable SkyValue value, @Nullable Iterable<SkyKey> directDeps) {
      FileOpNodeOrFuture result = getOrCreateValueForSubclasses(key);
      if (result instanceof FutureFileOpNode future) {
        if (future.tryTakeOwnership()) {
          try {
            return populateFutureFileOpNode(future, value, directDeps);
          } finally {
            future.verifyComplete();
          }
        }
      }
      return result;
    }
  }

  // Same method used by java.util.stream.AbstractTask.suggestTargetSize.
  private static final int TARGET_WORK_UNITS = Runtime.getRuntime().availableProcessors() * 4;

  private final SafeExecutor executor;
  private final InMemoryGraph graph;
  private final FileOpNodeMap nodes = new FileOpNodeMap();

  private ImmutableSet<SkyKey> selectedKeys;
  private boolean shouldDiscardMemory;
  @Nullable // non-null if shouldDiscardMemory is true
  private ImmutableSet<PackageIdentifier> referencedPackages;

  FileOpNodeMemoizingLookup(
      SafeExecutor executor,
      InMemoryGraph graph,
      ImmutableSet<SkyKey> selectedKeys,
      boolean shouldDiscardMemory,
      @Nullable ImmutableSet<PackageIdentifier> referencedPackages) {
    this.executor = executor;
    this.graph = graph;
    this.selectedKeys = selectedKeys;
    this.shouldDiscardMemory = shouldDiscardMemory;
    this.referencedPackages = referencedPackages;
  }

  void setMemoryReclamationParameters(
      ImmutableSet<SkyKey> selectedKeys,
      boolean shouldDiscardMemory,
      @Nullable ImmutableSet<PackageIdentifier> referencedPackages) {
    this.selectedKeys = selectedKeys;
    this.shouldDiscardMemory = shouldDiscardMemory;
    this.referencedPackages = referencedPackages;
  }

  FileOpNodeOrFuture computeNode(ActionLookupKey key) {
    return nodes.getValueOrFuture(key);
  }

  /**
   * Computes a node with the specified direct deps and value.
   *
   * <p>To be used when the node in question hasn't been committed to Skyframe yet.
   *
   * @param key the {@link SkyKey} for which the node should be computed
   * @param value the value that will be eventually committed to Skyframe under the specified key
   * @param directDeps the deps of the corresponding Skyframe node. Must be the same as what will be
   *     committed to Skyframe.
   */
  FileOpNodeOrFuture computeNode(ActionLookupKey key, SkyValue value, Iterable<SkyKey> directDeps) {
    return nodes.getValueOrFuture(key, value, directDeps);
  }

  private FileOpNodeOrFuture populateFutureFileOpNode(
      FutureFileOpNode ownedFuture,
      @Nullable SkyValue value,
      @Nullable Iterable<SkyKey> directDeps) {
    SkyKey key = ownedFuture.key();
    var collector = new FileOpNodeCollector(key);

    accumulateTransitiveFileSystemOperations(collector, key, value, directDeps);
    collector.finishRegistration();

    if (collector.isDone()) {
      try {
        return ownedFuture.completeWith(Futures.getDone(collector));
      } catch (ExecutionException e) {
        // Unwraps the ExecutionException transport envelope so ownedFuture fails with the root
        // cause. Failing with `e` directly would produce nested layers of ExecutionException across
        // future-to-future hops.
        Throwable cause = e.getCause();
        return ownedFuture.failWith(cause != null ? cause : e);
      } catch (CancellationException e) {
        ownedFuture.cancel(/* mayInterruptIfRunning= */ false);
        return ownedFuture;
      }
    }
    return ownedFuture.completeWith(collector);
  }

  private void accumulateTransitiveFileSystemOperations(
      FileOpNodeCollector collector,
      SkyKey key,
      @Nullable SkyValue value,
      @Nullable Iterable<SkyKey> directDeps) {
    if (directDeps == null) {
      InMemoryNodeEntry nodeEntry = graph.getIfPresent(key);
      if (nodeEntry == null) {
        collector.failWith(new MissingSkyframeEntryException(key));
        return;
      }
      directDeps = nodeEntry.getDirectDeps();
      value = checkNotNull(nodeEntry.getValue(), key);
    }

    if (key instanceof ActionLookupKey) {
      // If the corresponding value is an InputFileConfiguredTarget, it indicates an execution time
      // file dependency.
      if (value instanceof NonRuleConfiguredTargetValue nonRuleConfiguredTargetValue
          && nonRuleConfiguredTargetValue.getConfiguredTarget()
              instanceof InputFileConfiguredTarget inputFileConfiguredTarget) {
        // The source artifact's file becomes an execution time dependency of actions owned by
        // configured targets with this InputFileConfiguredTarget as a dependency.
        SourceArtifact source = inputFileConfiguredTarget.getArtifact();
        var fileKey = FileKey.create(source.getRootedPath());
        for (SkyKey dep : directDeps) {
          if (dep.equals(fileKey)) {
            continue;
          }
          switch (dep) {
            case FileOpNode immediateNode -> collector.addNode(immediateNode);
            default -> addNodeForKey(dep, collector);
          }
        }

        collector.setSource(fileKey);
        return;
      }
    }

    for (SkyKey dep : directDeps) {
      switch (dep) {
        case FileOpNode immediateNode -> collector.addNode(immediateNode);
        default -> addNodeForKey(dep, collector);
      }
    }
  }

  private void addNodeForKey(SkyKey key, FileOpNodeCollector collector) {
    // TODO: b/364831651 - This adds all traversed SkyKeys to `nodes`. Consider if certain types
    // should be excluded from memoization.

    // The file dependencies of action execution values are defined through the configured
    // targets defining their actions. We can skip traversing the action graph.
    SkyKey dependencyKey =
        switch (key) {
          case ActionLookupData lookupData -> lookupData.getActionLookupKey();
          case DerivedArtifact artifact -> artifact.getArtifactOwner();
          default -> key;
        };
    switch (nodes.getValueOrFuture(dependencyKey)) {
      case EMPTY_FILE_OP_NODE -> {}
      case FileOpNode node -> collector.addNode(node);
      // There is a graph made of futures that parallels the Skyframe dependency graph. Therefore,
      // it's a bad idea to use directExecutor() here because the amount of work that the completion
      // of the future unblocks can be quite large.
      case FutureFileOpNode future -> collector.addFuture(future, executor);
    }
  }

  private final class FileOpNodeCollector
      extends AccumulatingQuiescingFuture<FileOpNodeOrEmpty, FileOpNodeOrEmpty> {
    private final SkyKey key;
    private final Set<FileOpNode> nodes = ConcurrentHashMap.newKeySet();
    @Nullable private FileKey sourceFile = null;

    private FileOpNodeCollector(SkyKey key) {
      super(safeDirectExecutor());
      this.key = key;
    }

    @Override
    protected FileOpNodeOrEmpty getValue() {
      if (shouldDiscardMemory
          // PackageIdentifier keys (PackageValues) must not be discarded before any referencing
          // ConfiguredTarget is serialized. The ConfiguredTargetValueCodec uses the package to
          // obtain target information. They are cleaned up later using reference counting
          // after all selected targets that require them are uploaded.
          && !(key instanceof PackageIdentifier pkgId && referencedPackages.contains(pkgId))
          && !key.equals(BaselineOptionsFunction.BASELINE_CONFIGURATION.getKey())
          && !key.equals(BaselineOptionsFunction.BASELINE_EXEC_CONFIGURATION.getKey())
          && !selectedKeys.contains(key)) {
        graph.removeIfDone(key);
      }
      return AbstractNestedFileOpNodes.from(nodes, sourceFile);
    }

    private void addNode(FileOpNode node) {
      nodes.add(node);
    }

    private void setSource(FileKey sourceFile) {
      checkState(
          this.sourceFile == null,
          "Attempted to set source to %s but source already set to %s.",
          sourceFile,
          this.sourceFile);
      this.sourceFile = sourceFile;
    }

    private void failWith(MissingSkyframeEntryException e) {
      recordException(e);
    }

    @Override
    protected void accumulateFutureResult(FileOpNodeOrEmpty nodeOrEmpty) {
      switch (nodeOrEmpty) {
        case EMPTY_FILE_OP_NODE -> {}
        case FileOpNode node -> addNode(node);
      }
    }
  }

  public void registerRemoteFingerprint(SkyKey key, ByteString fingerprint) {
    FileOpNodeOrFuture old = nodes.put(key, new RemoteFileOpNode(fingerprint));
    // Control only gets here for nodes that have been freshly downloaded. This means that there
    // must be no corresponding entry in the map.
    Verify.verify(old == null);
  }

  static ActionLookupKey getDependencyKey(SkyKey key) {
    return switch (key) {
      case ActionLookupKey alk -> alk;
      case ActionLookupData ald -> ald.getActionLookupKey();
      case DerivedArtifact artifact -> artifact.getArtifactOwner();
      case ActionLookupSummaryKey alsk -> alsk.argument();
      default -> throw new IllegalStateException("unexpected key: " + key.getCanonicalName());
    };
  }

  /**
   * Materializes the node graph for every key in {@code selection}.
   *
   * <p>Failures do not end materialization early. This returns only once every dispatched batch and
   * every node future it reached has settled, so no work remains in flight afterwards.
   *
   * @return all node resolution and evaluation failures accumulated across the graph during
   *     materialization, or an empty list if graph materialization succeeded completely
   */
  @CheckReturnValue
  public ImmutableList<Throwable> materializeNodeGraph(ImmutableSet<SkyKey> selection)
      throws InterruptedException {
    // Parallel root triggering with cached-node filtering.
    ImmutableList<SkyKey> keys = selection.asList();
    var quiescence = new LookupQuiescence(keys);
    int totalKeys = keys.size();
    int batchSize = max(1, totalKeys / TARGET_WORK_UNITS);

    for (int start = 0; start < totalKeys; start += batchSize) {
      executor.execute(quiescence.newBatch(start, min(start + batchSize, totalKeys)));
    }

    // Release the initial pre-increment and wait for global DAG quiescence.
    quiescence.finishRegistration();
    try {
      return quiescence.get();
    } catch (ExecutionException | CancellationException e) {
      // LookupQuiescence collects errors instead of failing, and is never exposed for cancellation.
      throw new AssertionError("LookupQuiescence unexpectedly failed", e);
    }
  }

  /** Whether serialization skips {@code key}: absent from the graph, or deserialized from cache. */
  private boolean isSkipped(SkyKey key) {
    InMemoryNodeEntry entry = graph.getIfPresent(key);
    return entry == null || entry.getValue() instanceof DeserializedSkyValue;
  }

  /**
   * Waits for root dispatch and every node future it reaches to settle, collecting failures.
   *
   * <p>Unlike {@link QuiescingFuture#executeSubtask}, a failure here never completes the future
   * early, so {@link #get} returns only once no work remains in flight.
   */
  private final class LookupQuiescence extends QuiescingFuture<ImmutableList<Throwable>>
      implements FutureCallback<FileOpNodeOrEmpty> {
    // Soft bound on the number of errors that we accumlate here. This bound can be exceeded due to
    // a TOCTOU race condition, but it's of no practical concern.
    private static final int MAX_ERRORS = 100;

    private final ImmutableList<SkyKey> keys;
    private final Set<Throwable> errors = Sets.newConcurrentHashSet();

    private LookupQuiescence(ImmutableList<SkyKey> keys) {
      // Note that this superclass constructor initializes taskCount to 1. That protects against
      // premature completion while the parallel root loop is still registering tasks.
      super(safeDirectExecutor());
      this.keys = keys;
    }

    @Override
    protected ImmutableList<Throwable> getValue() {
      return ImmutableList.copyOf(errors);
    }

    /**
     * Creates a task visiting {@code keys[begin, limit)}, holding quiescence open until it ends.
     */
    private Batch newBatch(int begin, int limit) {
      return new Batch(begin, limit);
    }

    private final class Batch implements RejectionHandlingRunnable {
      private final int begin;
      private final int limit;

      private Batch(int begin, int limit) {
        this.begin = begin;
        this.limit = limit;
        increment();
      }

      @Override
      public void run() {
        try {
          for (int i = begin; i < limit; i++) {
            SkyKey key = keys.get(i);
            // Skip keys that are never uploaded (see isSkipped). Materializing their nodes is
            // wasted work and, for keys absent from the graph, would report a spurious
            // MissingSkyframeEntryException.
            if (isSkipped(key)) {
              continue;
            }

            ActionLookupKey depKey = getDependencyKey(key);
            switch (computeNode(depKey)) {
              // quiescence tracks completion of node dependencies.
              case FutureFileOpNode future -> track(future);
              // Nothing needs to be done for already complete nodes.
              case FileOpNodeOrEmpty _ -> {}
            }
          }
        } catch (Throwable t) {
          if (errors.size() < MAX_ERRORS) {
            errors.add(t);
          }
        } finally {
          decrement();
        }
      }

      @Override
      public void handleRejection(Throwable t) {
        if (errors.size() < MAX_ERRORS) {
          errors.add(t);
        }
        decrement();
      }
    }

    private void track(FutureFileOpNode future) {
      increment();
      Futures.addCallback(future, this, directExecutor());
    }

    @Override
    @DoNotCall("Only called via track")
    public void onSuccess(FileOpNodeOrEmpty unused) {
      decrement();
    }

    @Override
    @DoNotCall("Only called via track")
    public void onFailure(Throwable t) {
      errors.add(t);
      decrement();
    }
  }
}
