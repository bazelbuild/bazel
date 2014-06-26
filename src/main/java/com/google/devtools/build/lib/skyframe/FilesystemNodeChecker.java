// Copyright 2014 Google Inc. All rights reserved.
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
package com.google.devtools.build.lib.skyframe;

import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.base.Supplier;
import com.google.common.base.Suppliers;
import com.google.common.base.Throwables;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterables;
import com.google.common.collect.Lists;
import com.google.common.collect.Sets;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.concurrent.ExecutorShutdownUtil;
import com.google.devtools.build.lib.concurrent.Sharder;
import com.google.devtools.build.lib.concurrent.ThrowableRecordingRunnableWrapper;
import com.google.devtools.build.lib.util.LoggingUtil;
import com.google.devtools.build.lib.util.Pair;
import com.google.devtools.build.lib.util.io.TimestampGranularityMonitor;
import com.google.devtools.build.lib.vfs.BatchStat;
import com.google.devtools.build.lib.vfs.FileStatusWithDigest;
import com.google.devtools.build.lib.vfs.RootedPath;
import com.google.devtools.build.skyframe.AutoUpdatingGraph;
import com.google.devtools.build.skyframe.Node;
import com.google.devtools.build.skyframe.NodeKey;
import com.google.devtools.build.skyframe.NodeType;

import java.io.IOException;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Level;
import java.util.logging.Logger;

import javax.annotation.Nullable;

/**
 * A helper class to find dirty nodes by accessing the filesystem directly (contrast with
 * {@link DiffAwareness}).
 */
class FilesystemNodeChecker {

  private static final int DIRTINESS_CHECK_THREADS = 50;
  private static final Logger LOG = Logger.getLogger(FilesystemNodeChecker.class.getName());

  private static final Predicate<NodeKey> FILE_STATE_AND_DIRECTORY_LISTING_FILTER =
      NodeType.nodeTypeIsIn(ImmutableSet.of(NodeTypes.FILE_STATE, NodeTypes.DIRECTORY_LISTING));
  private static final Predicate<NodeKey> ACTION_FILTER =
      NodeType.nodeTypeIs(NodeTypes.ACTION_EXECUTION);

  private final TimestampGranularityMonitor tsgm;
  private final Supplier<Map<NodeKey, Node>> graphNodesSupplier;
  private AtomicInteger modifiedOutputFilesCounter = new AtomicInteger(0);

  FilesystemNodeChecker(final AutoUpdatingGraph graph, TimestampGranularityMonitor tsgm) {
    this.tsgm = tsgm;

    // Construct the full map view of the entire graph at most once ("memoized"), lazily. If
    // getDirtyFilesystemNodes(Iterable<NodeKey>) is called on an empty Iterable, we avoid having
    // to create the Map of node keys to values. This is useful in the case where the graph
    // getNodes() method could be slow.
    this.graphNodesSupplier = Suppliers.memoize(new Supplier<Map<NodeKey, Node>>() {
      @Override
      public Map<NodeKey, Node> get() {
        return graph.getNodes();
      }
    });
  }

  Iterable<NodeKey> getFilesystemNodeKeys() {
    return Iterables.filter(graphNodesSupplier.get().keySet(),
        FILE_STATE_AND_DIRECTORY_LISTING_FILTER);
  }

  Collection<NodeKey> getDirtyFilesystemNodeKeys() throws InterruptedException {
    return getDirtyFilesystemNodes(getFilesystemNodeKeys());
  }

  /**
   * Check the given file and directory nodes for modifications. {@code nodes} is assumed to only
   * have {@link FileNode}s and {@link DirectoryListingNode}s.
   */
  Collection<NodeKey> getDirtyFilesystemNodes(Iterable<NodeKey> nodes)
      throws InterruptedException {
    return getDirtyNodes(nodes, FILE_STATE_AND_DIRECTORY_LISTING_FILTER, new DirtyChecker() {
      @Override
      public boolean isDirty(NodeKey key, Node value, TimestampGranularityMonitor tsgm) {
        return ((key.getNodeType() == NodeTypes.FILE_STATE
            && fileStateNodeIsDirty((RootedPath) key.getNodeName(), (FileStateNode) value, tsgm))
            || (key.getNodeType() == NodeTypes.DIRECTORY_LISTING
            && directoryNodeIsDirty((DirectoryListingNode) value)));
      }
    });
  }

  /**
   * Return a collection of action nodes which have output files that are not in-sync with
   * the on-disk file value (were modified externally).
   */
  public Collection<NodeKey> getDirtyActionNodes(@Nullable final BatchStat batchStatter)
      throws InterruptedException {
    // CPU-bound (usually) stat() calls, plus a fudge factor.
    LOG.info("Accumulating dirty actions");
    final int numOutputJobs = Runtime.getRuntime().availableProcessors() * 4;
    final Set<NodeKey> actionNodeKeys =
        Sets.filter(graphNodesSupplier.get().keySet(), ACTION_FILTER);
    final Sharder<Pair<NodeKey, ActionExecutionNode>> outputShards =
        new Sharder<>(numOutputJobs, actionNodeKeys.size());

    for (NodeKey key : actionNodeKeys) {
      outputShards.add(Pair.of(key, (ActionExecutionNode) graphNodesSupplier.get().get(key)));
    }
    LOG.info("Sharded action nodes for batching");

    ExecutorService executor = Executors.newFixedThreadPool(
        numOutputJobs,
        new ThreadFactoryBuilder().setNameFormat("FileSystem Output File Invalidator %d").build());

    Collection<NodeKey> dirtyKeys = Sets.newConcurrentHashSet();
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FileSystemNodeChecker#getDirtyActionNodes");

    modifiedOutputFilesCounter.set(0);
    for (List<Pair<NodeKey, ActionExecutionNode>> shard : outputShards) {
      Runnable job = (batchStatter == null)
          ? outputStatJob(dirtyKeys, shard)
          : batchStatJob(dirtyKeys, shard, batchStatter);
      executor.submit(wrapper.wrap(job));
    }

    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    LOG.info("Completed output file stat checks");
    if (interrupted) {
      throw new InterruptedException();
    }
    return dirtyKeys;
  }

  private Runnable batchStatJob(final Collection<NodeKey> dirtyKeys,
                                       final List<Pair<NodeKey, ActionExecutionNode>> shard,
                                       final BatchStat batchStatter) {
    return new Runnable() {
      @Override
      public void run() {
        Map<Artifact, Pair<NodeKey, ActionExecutionNode>> artifactToKeyAndValue = new HashMap<>();
        for (Pair<NodeKey, ActionExecutionNode> keyAndValue : shard) {
          ActionExecutionNode actionNode = keyAndValue.getSecond();
          if (actionNode == null) {
            dirtyKeys.add(keyAndValue.getFirst());
          } else {
            for (Artifact artifact : actionNode.getAllOutputArtifactData().keySet()) {
              artifactToKeyAndValue.put(artifact, keyAndValue);
            }
          }
        }

        List<Artifact> artifacts = ImmutableList.copyOf(artifactToKeyAndValue.keySet());
        List<FileStatusWithDigest> stats;
        try {
          stats = batchStatter.batchStat(/*includeDigest=*/true, /*includeLinks=*/true,
                                         Artifact.asPathFragments(artifacts));
        } catch (IOException e) {
          // Batch stat did not work. Log an exception and fall back on system calls.
          LoggingUtil.logToRemote(Level.WARNING, "Unable to process batch stat", e);
          outputStatJob(dirtyKeys, shard).run();
          return;
        } catch (InterruptedException e) {
          // We handle interrupt in the main thread.
          return;
        }

        Preconditions.checkState(artifacts.size() == stats.size(),
            "artifacts.size() == %s stats.size() == %s", artifacts.size(), stats.size());
        for (int i = 0; i < artifacts.size(); i++) {
          Artifact artifact = artifacts.get(i);
          FileStatusWithDigest stat = stats.get(i);
          Pair<NodeKey, ActionExecutionNode> keyAndValue = artifactToKeyAndValue.get(artifact);
          ActionExecutionNode actionNode = keyAndValue.getSecond();
          NodeKey key = keyAndValue.getFirst();
          FileNode lastKnownData = actionNode.getAllOutputArtifactData().get(artifact);
          try {
            FileNode newData = FileAndMetadataCache.fileNodeFromArtifact(artifact, stat, tsgm);
            if (!newData.equals(lastKnownData)) {
              modifiedOutputFilesCounter.getAndIncrement();
              dirtyKeys.add(key);
            }
          } catch (IOException e) {
            // This is an unexpected failure getting a digest or symlink target.
            modifiedOutputFilesCounter.getAndIncrement();
            dirtyKeys.add(key);
          }
        }
      }
    };
  }

  private Runnable outputStatJob(final Collection<NodeKey> dirtyKeys,
                                 final List<Pair<NodeKey, ActionExecutionNode>> shard) {
    return new Runnable() {
      @Override
      public void run() {
        for (Pair<NodeKey, ActionExecutionNode> keyAndValue : shard) {
          ActionExecutionNode node = keyAndValue.getSecond();
          if (node == null || actionNodeIsDirtyWithDirectSystemCalls(node)) {
            dirtyKeys.add(keyAndValue.getFirst());
          }
        }
      }
    };
  }

  /**
   * Returns number of modified output files inside of dirty actions.
   */
  int getNumberOfModifiedOutputFiles() {
    return modifiedOutputFilesCounter.get();
  }

  private boolean actionNodeIsDirtyWithDirectSystemCalls(ActionExecutionNode actionNode) {
    boolean isDirty = false;
    for (Map.Entry<Artifact, FileNode> entry :
        actionNode.getAllOutputArtifactData().entrySet()) {
      Artifact artifact = entry.getKey();
      FileNode lastKnownData = entry.getValue();
      try {
        if (!FileAndMetadataCache.fileNodeFromArtifact(artifact, null, tsgm).equals(
            lastKnownData)) {
          modifiedOutputFilesCounter.getAndIncrement();
          isDirty = true;
        }
      } catch (IOException e) {
        // This is an unexpected failure getting a digest or symlink target.
        modifiedOutputFilesCounter.getAndIncrement();
        isDirty = true;
      }
    }
    return isDirty;
  }

  private Collection<NodeKey> getDirtyNodes(Iterable<NodeKey> nodes,
                                            Predicate<NodeKey> keyFilter,
                                            final DirtyChecker checker)
      throws InterruptedException {
    ExecutorService executor = Executors.newFixedThreadPool(DIRTINESS_CHECK_THREADS,
        new ThreadFactoryBuilder().setNameFormat("FileSystem Node Invalidator %d").build());

    Collection<NodeKey> dirtyKeys = Lists.newArrayList();
    final Collection<NodeKey> concurrentDirtyKeys = Collections.synchronizedCollection(dirtyKeys);
    ThrowableRecordingRunnableWrapper wrapper =
        new ThrowableRecordingRunnableWrapper("FilesystemNodeChecker#getDirtyNodes");
    for (final NodeKey key : nodes) {
      Preconditions.checkState(keyFilter.apply(key), key);
      final Node value = graphNodesSupplier.get().get(key);
      executor.execute(wrapper.wrap(new Runnable() {
        @Override
        public void run() {
          // value will be null if the node is in error or part of a cycle.
          // TODO(bazel-team): This is overly conservative.
          if (value == null || checker.isDirty(key, value, tsgm)) {
            concurrentDirtyKeys.add(key);
          }
        }
      }));
    }

    boolean interrupted = ExecutorShutdownUtil.interruptibleShutdown(executor);
    Throwables.propagateIfPossible(wrapper.getFirstThrownError());
    if (interrupted) {
      throw new InterruptedException();
    }
    return dirtyKeys;
  }

  private static boolean fileStateNodeIsDirty(RootedPath rootedPath, FileStateNode fileStateNode,
      TimestampGranularityMonitor tsgm) {
    try {
      FileStateNode newNode = FileStateNode.create(rootedPath, tsgm);
      return !newNode.equals(fileStateNode);
    } catch (InconsistentFilesystemException | IOException e) {
      // TODO(bazel-team): An IOException indicates a failure to get a file digest or a symlink
      // target, not a missing file. Such a failure really shouldn't happen, so failing early
      // may be better here.
      return true;
    }
  }

  private static boolean directoryNodeIsDirty(DirectoryListingNode directoryNode) {
    try {
      DirectoryListingNode newNode = DirectoryListingNode.nodeForRootedPath(
          directoryNode.getRootedPath());
      return !newNode.equals(directoryNode);
    } catch (IOException e) {
      return true;
    }
  }

  private interface DirtyChecker {
    boolean isDirty(NodeKey key, Node value, TimestampGranularityMonitor tsgm);
  }
}
