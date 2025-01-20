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
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;

import com.google.common.util.concurrent.Futures;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.Artifact.SourceArtifact;
import com.google.devtools.build.lib.analysis.configuredtargets.InputFileConfiguredTarget;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FutureFileOpNode;
import com.google.devtools.build.lib.skyframe.NonRuleConfiguredTargetValue;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.InMemoryGraph;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutionException;

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
final class FileOpNodeMemoizingLookup {
  private final InMemoryGraph graph;

  private final ValueOrFutureMap<SkyKey, FileOpNodeOrFuture, FileOpNodeOrEmpty, FutureFileOpNode>
      nodes =
          new ValueOrFutureMap<>(
              new ConcurrentHashMap<>(),
              FutureFileOpNode::new,
              this::populateFutureFileOpNode,
              FutureFileOpNode.class);

  FileOpNodeMemoizingLookup(InMemoryGraph graph) {
    this.graph = graph;
  }

  FileOpNodeOrFuture computeNode(ActionLookupKey key) {
    return nodes.getValueOrFuture(key);
  }

  private FileOpNodeOrFuture populateFutureFileOpNode(FutureFileOpNode ownedFuture) {
    var builder = new FileOpNodeBuilder();

    accumulateTransitiveFileSystemOperations(ownedFuture.key(), builder);

    if (!builder.hasFutures()) {
      // This can be empty for certain functions, e.g., PRECOMPUTED, IGNORED_PACKAGE_PREFIXES and
      // PARSED_FLAGS.
      return ownedFuture.completeWith(builder.call());
    }
    return ownedFuture.completeWith(
        Futures.whenAllSucceed(builder.futureNodes).call(builder, directExecutor()));
  }

  private void accumulateTransitiveFileSystemOperations(SkyKey key, FileOpNodeBuilder builder) {
    for (SkyKey dep : checkNotNull(graph.getIfPresent(key), key).getDirectDeps()) {
      switch (dep) {
        case FileOpNode immediateNode:
          builder.addNode(immediateNode);
          break;
        default:
          addNodeForKey(dep, builder);
          break;
      }
    }
  }

  private void addNodeForKey(SkyKey key, FileOpNodeBuilder builder) {
    if (key instanceof ActionLookupKey actionLookupKey) {
      var nodeEntry = checkNotNull(graph.getIfPresent(key), key);
      // If the corresponding value is an InputFileConfiguredTarget, it indicates an execution time
      // file dependency.
      if ((checkNotNull(nodeEntry.getValue(), actionLookupKey)
              instanceof NonRuleConfiguredTargetValue nonRuleConfiguredTargetValue)
          && (nonRuleConfiguredTargetValue.getConfiguredTarget()
              instanceof InputFileConfiguredTarget inputFileConfiguredTarget)) {
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
          builder.addSource(fileKey);
        }
      }
    }

    // TODO: b/364831651 - This adds all traversed SkyKeys to `nodes`. Consider if certain types
    // should be excluded from memoization.
    switch (nodes.getValueOrFuture(key)) {
      case EMPTY_FILE_OP_NODE:
        break;
      case FileOpNode node:
        builder.addNode(node);
        break;
      case FutureFileOpNode future:
        builder.addFuture(future);
        break;
    }
  }

  private static class FileOpNodeBuilder implements Callable<FileOpNodeOrEmpty> {
    private final HashSet<FileOpNode> nodes = new HashSet<>();

    private final HashSet<FileKey> sourceFiles = new HashSet<>();

    private final ArrayList<FutureFileOpNode> futureNodes = new ArrayList<>();

    /** Called only after all futures in {@link #futureNodes} succeed. */
    @Override
    public FileOpNodeOrEmpty call() {
      for (FutureFileOpNode future : futureNodes) {
        try {
          addNode(Futures.getDone(future));
        } catch (ExecutionException e) {
          throw new IllegalStateException(
              "unexpected exception, should only be called after success", e);
        }
      }
      return AbstractNestedFileOpNodes.from(nodes, sourceFiles);
    }

    private boolean hasFutures() {
      return !futureNodes.isEmpty();
    }

    private void addNode(FileOpNodeOrEmpty nodeOrEmpty) {
      switch (nodeOrEmpty) {
        case EMPTY_FILE_OP_NODE:
          break;
        case FileOpNode node:
          nodes.add(node);
          break;
      }
    }

    private void addSource(FileKey sourceFile) {
      sourceFiles.add(sourceFile);
    }

    private void addFuture(FutureFileOpNode future) {
      futureNodes.add(future);
    }
  }
}
