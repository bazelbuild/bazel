// Copyright 2026 The Bazel Authors. All rights reserved.
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

import static com.google.common.base.Preconditions.checkState;
import static com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.EmptyFileOpNode.EMPTY_FILE_OP_NODE;

import com.google.common.collect.ImmutableList;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode;
import com.google.devtools.build.skyframe.SkyKey;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * The selection, split for the leaf walk and for upload ordering in a single serial pass.
 *
 * <p>This pass is where each selected key is translated to the {@link ActionLookupKey} that owns
 * its file dependencies, so it is also where analysis keys and owned keys part ways. Three buckets,
 * emitted in this order:
 *
 * <ol>
 *   <li><b>Unranked analysis keys</b>, whose node is anything other than an {@link
 *       AbstractNestedFileOpNodes}: empty, a {@link RemoteFileOpNode}, or a bare leaf. A bare leaf
 *       is a single {@link FileKey} or {@link DirectoryListingKey} serving as the key's entire
 *       node: {@link AbstractNestedFileOpNodes#from} returns the sole element of a one-element
 *       collection rather than wrapping it, so a key with no source whose only collected dependency
 *       is such a leaf has that leaf as its node. The walk assigns these keys no depth. They go
 *       first, and that is sound across the boundary: an analysis key with an empty node can only
 *       depend on others with empty nodes, since any file operation below it would make its own
 *       node non-empty; and one whose node is a bare leaf can only depend on empty nodes or the
 *       same leaf, since a nested child would make its own node nested or collapse onto that child.
 *       Nothing here depends on a ranked key.
 *   <li><b>Ranked analysis keys</b>, ordered by the longest-path depth of their node, deepest first
 *       (see {@link AbstractNestedFileOpNodes#getLongestPathDepth}). If one depends on another, the
 *       dependency's node is in the dependent's collector, so it is either a child at a strictly
 *       greater depth or, when it was the only element, the very node the dependent collapsed onto.
 *       Keys sharing a node therefore tie, as do unrelated keys at the same depth; ties keep
 *       selection order.
 *   <li><b>Owned keys</b>, in selection order. These are the keys that are not themselves {@link
 *       ActionLookupKey}s and so take their file dependencies from an owner (see {@link
 *       FileOpNodeMemoizingLookup#getDependencyKey}): execution keys ({@link ActionLookupData} and
 *       {@link DerivedArtifact}) and {@link ActionLookupSummaryKey}s. Their serialized values are
 *       small and share little with each other, so their relative order does not matter, and
 *       appending them after every analysis key places each after its owner.
 * </ol>
 *
 * <p>The only inversions this admits are within the unranked bucket and among ties.
 *
 * <p>Ordering by depth rather than by post-order schedules each node as late as its dependents
 * allow. Every node below the top depth has a parent exactly one level shallower, so its bitmap is
 * consumed soon after it is produced rather than whenever the depth-first walk happened to return
 * to a parent. That shortens bitmap lifetimes, and with them peak residency.
 *
 * <p>Owned keys still contribute their owner's node to {@link #roots}. That keeps the walk's input,
 * and so the leaf numbering, identical to what it was before upload ordering was derived from it.
 *
 * <p>Selection order is not deterministic, so neither is the order among ties or within the
 * unranked and owned buckets. Upload order only affects scheduling, and so when memory can be
 * reclaimed. It does not affect what gets deduplicated or the structure of the resulting
 * invalidation data.
 */
final class SelectionPartition {
  /** Nodes for the leaf walk, in selection order. Duplicates are retained; the walk dedups. */
  private final ArrayList<FileOpNode> roots = new ArrayList<>();

  private final ArrayList<ActionLookupKey> unrankedAnalysisKeys = new ArrayList<>();

  /** Parallel to {@link #rankedNodes}. */
  private final ArrayList<ActionLookupKey> rankedAnalysisKeys = new ArrayList<>();

  private final ArrayList<AbstractNestedFileOpNodes> rankedNodes = new ArrayList<>();

  /** Execution keys and {@link ActionLookupSummaryKey}s; see the class comment. */
  private final ArrayList<SkyKey> ownedKeys = new ArrayList<>();

  private SelectionPartition() {}

  /**
   * Partitions {@code selection}.
   *
   * <p>Keys that serialization skips are left out. Keys whose value was deserialized from cache
   * match the early return in {@link SelectedEntrySerializer#upload}. Keys absent from the graph
   * are dropped from the upload order altogether, since {@code upload} would throw for them.
   *
   * @param isSkipped keys that serialization skips, which are left out of the partition entirely
   * @param resolveNode the already materialized node of an {@link ActionLookupKey}. Called at most
   *     once per distinct owner.
   */
  static SelectionPartition partition(
      Iterable<? extends SkyKey> selection,
      Predicate<SkyKey> isSkipped,
      Function<ActionLookupKey, FileOpNodeOrEmpty> resolveNode) {
    var partition = new SelectionPartition();
    var resolvedNodes = new HashMap<ActionLookupKey, FileOpNodeOrEmpty>();
    for (SkyKey key : selection) {
      if (isSkipped.test(key)) {
        continue;
      }
      partition.add(
          key,
          resolvedNodes.computeIfAbsent(
              FileOpNodeMemoizingLookup.getDependencyKey(key), resolveNode));
    }
    return partition;
  }

  /** Nodes for the leaf walk, in selection order. Duplicates are retained; the walk dedups. */
  List<FileOpNode> roots() {
    return Collections.unmodifiableList(roots);
  }

  private void add(SkyKey key, FileOpNodeOrEmpty node) {
    switch (node) {
      case FileOpNode fileOpNode -> roots.add(fileOpNode);
      case EMPTY_FILE_OP_NODE -> {}
    }
    if (key instanceof ActionLookupKey lookupKey) {
      if (node instanceof AbstractNestedFileOpNodes nested) {
        rankedAnalysisKeys.add(lookupKey);
        rankedNodes.add(nested);
      } else {
        unrankedAnalysisKeys.add(lookupKey);
      }
    } else {
      // An execution key or an ActionLookupSummaryKey; node is its owner's.
      ownedKeys.add(key);
    }
  }

  /**
   * Assembles the upload order. Must run after the leaf walk has assigned depths.
   *
   * <p>A counting sort: every ranked node's depth is a small integer in {@code [TOP_DEPTH,
   * maxDepth]}, so an array indexed by depth can hold one chain of keys per depth. Each ranked key
   * is chained onto its node's depth and the chains are read out from the deepest up. Chains are
   * built by prepending in reverse, so ties keep selection order.
   *
   * @param maxDepth the greatest depth the walk assigned to any node
   */
  ImmutableList<SkyKey> uploadOrder(int maxDepth) {
    int ranked = rankedAnalysisKeys.size();
    // head stores the index of the first key at the corresponding depth.
    // For example: if head[1] = 5 means that the rankedAnalysisKeys.get(5) is the first key at
    // depth 1
    int[] head = new int[maxDepth + 1];
    Arrays.fill(head, -1); // -1 is the end-of-list sentinel value
    // `next` stores next pointers for per-depth lists rooted in `head`.
    // For example: if head[1] = 5 and next[5] = 7, next[7] = -1, then depth 1 contains 5 and 7.
    int[] next = new int[ranked];
    for (int i = ranked - 1; i >= 0; i--) {
      int depth = rankedNodes.get(i).getLongestPathDepth();
      checkState(
          depth >= AbstractNestedFileOpNodes.TOP_DEPTH && depth <= maxDepth,
          "depth %s of %s is outside [%s, %s]: the node was not reached by this walk",
          depth,
          rankedAnalysisKeys.get(i),
          AbstractNestedFileOpNodes.TOP_DEPTH,
          maxDepth);
      next[i] = head[depth];
      head[depth] = i;
    }

    var order =
        ImmutableList.<SkyKey>builderWithExpectedSize(
            unrankedAnalysisKeys.size() + ranked + ownedKeys.size());
    order.addAll(unrankedAnalysisKeys);
    for (int depth = maxDepth; depth >= AbstractNestedFileOpNodes.TOP_DEPTH; depth--) {
      for (int i = head[depth]; i >= 0; i = next[i]) {
        order.add(rankedAnalysisKeys.get(i));
      }
    }
    return order.addAll(ownedKeys).build();
  }
}
