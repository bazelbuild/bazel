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
package com.google.devtools.build.lib.skyframe;

import static com.google.common.util.concurrent.MoreExecutors.directExecutor;
import static com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.NUM_JOBS;

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableMap;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.ListeningExecutorService;
import com.google.common.util.concurrent.MoreExecutors;
import com.google.common.util.concurrent.ThreadFactoryBuilder;
import com.google.devtools.build.lib.actions.ActionAnalysisMetadata;
import com.google.devtools.build.lib.actions.ActionLookupValue;
import com.google.devtools.build.lib.actions.Artifact;
import com.google.devtools.build.lib.actions.ArtifactPrefixConflictException;
import com.google.devtools.build.lib.actions.MutableActionGraph;
import com.google.devtools.build.lib.actions.MutableActionGraph.ActionConflictException;
import com.google.devtools.build.lib.concurrent.ExecutorUtil;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.profiler.SilentCloseable;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ActionConflictsAndStats;
import com.google.devtools.build.lib.skyframe.ArtifactConflictFinder.ConflictException;
import com.google.devtools.build.lib.vfs.PathFragment;
import com.google.devtools.build.skyframe.SkyValue;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.Executors;

/**
 * An incremental artifact conflict finder that maintains a running state.
 *
 * <p>Once an ActionLookupKey is analyzed, its actions are registered with this conflict finder
 * before execution. The internal action graph accumulates these actions in order to detect a
 * conflict later on. There should be one instance of this class per build.
 */
@ThreadSafe
public final class IncrementalArtifactConflictFinder {
  private final MutableActionGraph threadSafeMutableActionGraph;
  private final ConcurrentMap<String, Object> pathFragmentTrieRoot;
  private final ListeningExecutorService executorService;

  private IncrementalArtifactConflictFinder(MutableActionGraph threadSafeMutableActionGraph) {
    this.threadSafeMutableActionGraph = threadSafeMutableActionGraph;
    this.pathFragmentTrieRoot = new ConcurrentHashMap<>();
    this.executorService =
        MoreExecutors.listeningDecorator(
            Executors.newFixedThreadPool(
                NUM_JOBS,
                new ThreadFactoryBuilder()
                    .setNameFormat("IncrementalArtifactConflictFinder %d")
                    .build()));
  }

  static IncrementalArtifactConflictFinder createWithActionGraph(
      MutableActionGraph threadSafeMutableActionGraph) {
    return new IncrementalArtifactConflictFinder(threadSafeMutableActionGraph);
  }

  public int getOutputArtifactCount() {
    return threadSafeMutableActionGraph.getSize();
  }

  ActionConflictsAndStats findArtifactConflicts(
      ImmutableCollection<SkyValue> actionLookupValues, boolean strictConflictChecks)
      throws InterruptedException {
    ConcurrentMap<ActionAnalysisMetadata, ConflictException> temporaryBadActionMap =
        new ConcurrentHashMap<>();

    try (SilentCloseable c =
        Profiler.instance().profile("IncrementalArtifactConflictFinder.findArtifactConflicts")) {
      constructActionGraphAndArtifactList(
          pathFragmentTrieRoot,
          actionLookupValues,
          strictConflictChecks,
          temporaryBadActionMap);
    }

    return ActionConflictsAndStats.create(
        ImmutableMap.copyOf(temporaryBadActionMap), threadSafeMutableActionGraph.getSize());
  }

  private void constructActionGraphAndArtifactList(
      ConcurrentMap<String, Object> pathFragmentTrieRoot,
      ImmutableCollection<SkyValue> actionLookupValues,
      boolean strictConflictChecks,
      ConcurrentMap<ActionAnalysisMetadata, ConflictException> badActionMap)
      throws InterruptedException {
    List<ListenableFuture<Void>> futures = new ArrayList<>(actionLookupValues.size());
    synchronized (executorService) {
      // Some other thread shut down the executor, exit now.
      if (executorService.isShutdown()) {
        return;
      }
      for (SkyValue alv : actionLookupValues) {
        futures.add(
            executorService.submit(
                () ->
                    actionRegistration(
                        alv,
                        threadSafeMutableActionGraph,
                        pathFragmentTrieRoot,
                        strictConflictChecks,
                        badActionMap)));
      }
    }
    // Now wait on the futures.
    try {
      Futures.whenAllComplete(futures).call(() -> null, directExecutor()).get();
    } catch (ExecutionException e) {
      throw new IllegalStateException("Unexpected exception", e);
    }
  }

  void shutdown() {
    synchronized (executorService) {
      if (!executorService.isShutdown() && ExecutorUtil.interruptibleShutdown(executorService)) {
        // Preserve the interrupt status.
        Thread.currentThread().interrupt();
      }
    }
  }

  private static Void actionRegistration(
      SkyValue value,
      MutableActionGraph actionGraph,
      ConcurrentMap<String, Object> pathFragmentTrieRoot,
      boolean strictConflictChecks,
      ConcurrentMap<ActionAnalysisMetadata, ConflictException> badActionMap) {
    Preconditions.checkState(value instanceof ActionLookupValue);
    for (ActionAnalysisMetadata action : ((ActionLookupValue) value).getActions()) {
      try {
        actionGraph.registerAction(action);
      } catch (ActionConflictException e) {
        // It may be possible that we detect a conflict for the same action more than once, if
        // that action belongs to multiple aspect values. In this case we will harmlessly
        // overwrite the badActionMap entry.
        badActionMap.put(action, new ConflictException(e));
        // We skip the rest of the loop, and do not add the path->artifact mapping for this
        // artifact below -- we don't need to check it since this action is already in
        // error.
        continue;
      } catch (InterruptedException e) {
        // Bail.
        Thread.currentThread().interrupt();
        return null;
      }
      for (Artifact output : action.getOutputs()) {
        checkOutputPrefix(
            actionGraph, strictConflictChecks, pathFragmentTrieRoot, output, badActionMap);
      }
    }
    return null;
  }

  /**
   * Fits the path segments into the existing trie.
   *
   * <p>A conceptual path segment TrieNode can be:
   *
   * <ul>
   *   <li>an Artifact if it's a leaf node, or
   *   <li>a {@code ConcurrentMap<String, Object>} if it's a non-leaf node. The mapping is from a
   *       path segment to another trie node.
   * </ul>
   *
   * <p>We do this instead of creating a proper wrapper TrieNode data structure to save memory, as
   * the trie is expected to get quite large.
   */
  private static void checkOutputPrefix(
      MutableActionGraph actionGraph,
      boolean strictConflictCheck,
      ConcurrentMap<String, Object> root,
      Artifact newArtifact,
      ConcurrentMap<ActionAnalysisMetadata, ConflictException> badActionMap) {
    Object existingTrieNode = root;
    PathFragment newArtifactPathFragment = newArtifact.getExecPath();
    Iterator<String> newPathIter = newArtifactPathFragment.segments().iterator();

    while (newPathIter.hasNext() && !(existingTrieNode instanceof Artifact)) {
      String newSegment = newPathIter.next();
      boolean isFinalSegmentOfNewPath = !newPathIter.hasNext();
      @SuppressWarnings("unchecked")
      ConcurrentMap<String, Object> existingNonLeafNode =
          (ConcurrentMap<String, Object>) existingTrieNode;

      Object matchingChildNode =
          existingNonLeafNode.computeIfAbsent(
              newSegment,
              isFinalSegmentOfNewPath
                  ? unused -> newArtifact
                  : unused -> new ConcurrentHashMap<String, Object>());

      // By the time we arrive in this method, we know for sure that there can't be any exact
      // matches in the paths since that would have been an ActionConflictException.
      boolean newPathIsPrefixOfExisting =
          !(matchingChildNode instanceof Artifact) && isFinalSegmentOfNewPath;
      boolean existingPathIsPrefixOfNew =
          matchingChildNode instanceof Artifact && !isFinalSegmentOfNewPath;

      if (existingPathIsPrefixOfNew || newPathIsPrefixOfExisting) {
        Artifact conflictingExistingArtifact = getOwningArtifactFromTrie(matchingChildNode);
        ActionAnalysisMetadata priorAction =
            Preconditions.checkNotNull(
                actionGraph.getGeneratingAction(conflictingExistingArtifact),
                conflictingExistingArtifact);
        ActionAnalysisMetadata currentAction =
            Preconditions.checkNotNull(actionGraph.getGeneratingAction(newArtifact), newArtifact);
        if (strictConflictCheck || priorAction.shouldReportPathPrefixConflict(currentAction)) {
          ConflictException exception =
              new ConflictException(
                  new ArtifactPrefixConflictException(
                      conflictingExistingArtifact.getExecPath(),
                      newArtifactPathFragment,
                      priorAction.getOwner().getLabel(),
                      currentAction.getOwner().getLabel()));
          badActionMap.put(priorAction, exception);
          badActionMap.put(currentAction, exception);
        }
        // If 2 paths collide, we need to update the Trie to contain only the shorter one.
        // This is required for correctness: the set of subsequent paths that could conflict with
        // the longer path is a subset of that of the shorter path.
        if (newPathIsPrefixOfExisting) {
          existingNonLeafNode.put(newSegment, newArtifact);
        }

        break;
      }
      existingTrieNode = matchingChildNode;
    }
  }

  // TODO(b/214389062) Fix the issue with SolibSymlinkAction before launch.
  private static Artifact getOwningArtifactFromTrie(Object trieNode) {
    Preconditions.checkArgument(
        trieNode instanceof Artifact || trieNode instanceof ConcurrentHashMap);
    if (trieNode instanceof Artifact) {
      return (Artifact) trieNode;
    }
    Object nodeIter = trieNode;
    while (!(nodeIter instanceof Artifact)) {
      // Just pick the first path available down the Trie.
      for (Object value : ((ConcurrentHashMap<?, ?>) nodeIter).values()) {
        nodeIter = value;
        break;
      }
    }
    return (Artifact) nodeIter;
  }
}
