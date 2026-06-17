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
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;

import com.google.common.collect.ImmutableSet;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.cmdline.PackageIdentifier;
import com.google.devtools.build.lib.concurrent.QuiescingFuture;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.NonRuleConfiguredTargetValue;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.InMemoryNodeEntry;
import com.google.devtools.build.skyframe.SkyKey;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executor;
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

  private final Executor executor;
  private final InMemoryGraph graph;
  private final ImmutableSet<SkyKey> selectedKeys;
  private final boolean shouldDiscardMemory;
  @Nullable // non-null if shouldDiscardMemory is true
  private final ImmutableSet<PackageIdentifier> referencedPackages;

  private final FileOpNodeMap nodes = new FileOpNodeMap();

  FileOpNodeMemoizingLookup(
      Executor executor,
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
    var collector = new FileOpNodeCollector(executor, key);

    accumulateTransitiveFileSystemOperations(collector, key, value, directDeps);
    collector.notifyAllFuturesAdded();

    if (collector.isDone()) {
      try {
        return ownedFuture.completeWith(Futures.getDone(collector));
      } catch (ExecutionException e) {
        return ownedFuture.failWith(e);
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
        var fileKey =
            FileKey.create(RootedPath.toRootedPath(source.getRoot().getRoot(), source.getPath()));
        if (graph.getIfPresent(fileKey) != null) {
          // If the file value is not present in the graph, it means that no action executed
          // actually depended on that file.
          //
          // TODO: b/364831651 - for greater determinism, consider performing additional Skyframe
          // evaluations for these unused dependencies.
          collector.setSource(fileKey);
        }
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
      case FutureFileOpNode future -> collector.addFuture(future);
    }
  }

  private final class FileOpNodeCollector extends QuiescingFuture<FileOpNodeOrEmpty>
      implements FutureCallback<FileOpNodeOrEmpty> {
    private final Executor executor;
    private final SkyKey key;
    private final Set<FileOpNode> nodes = ConcurrentHashMap.newKeySet();
    @Nullable private FileKey sourceFile = null;

    private FileOpNodeCollector(Executor executor, SkyKey key) {
      super(directExecutor());
      this.executor = executor;
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

    private void addFuture(FutureFileOpNode future) {
      increment();
      // There is a graph made of futures that parallels the Skyframe dependency graph. Therefore,
      // it's a bad idea to use directExecutor() here because the amount of work that the
      // the completion of the future unblocks can be quite large.
      Futures.addCallback(future, (FutureCallback<FileOpNodeOrEmpty>) this, executor);
    }

    private void notifyAllFuturesAdded() {
      decrement();
    }

    private void failWith(MissingSkyframeEntryException e) {
      notifyException(e);
    }

    /**
     * Implementation of {@link FutureCallback<FileOpNode>}.
     *
     * @deprecated do not call, only used for callback processing
     */
    @Deprecated
    @Override
    public void onSuccess(FileOpNodeOrEmpty nodeOrEmpty) {
      switch (nodeOrEmpty) {
        case EMPTY_FILE_OP_NODE -> {}
        case FileOpNode node -> addNode(node);
      }
      decrement();
    }

    /**
     * Implementation of {@link FutureCallback<FileOpNode>}.
     *
     * @deprecated do not call, only used for callback processing
     */
    @Deprecated
    @Override
    public void onFailure(Throwable t) {
      notifyException(t);
    }
  }
}
