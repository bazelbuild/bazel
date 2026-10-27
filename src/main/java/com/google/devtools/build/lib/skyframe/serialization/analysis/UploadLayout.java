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
import static java.lang.Math.max;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.actions.ActionLookupData;
import com.google.devtools.build.lib.actions.ActionLookupKey;
import com.google.devtools.build.lib.actions.ActionLookupSummaryKey;
import com.google.devtools.build.lib.actions.Artifact.DerivedArtifact;
import com.google.devtools.build.lib.profiler.Profiler;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes.NestedFileOpNodesWithSource;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNodeOrEmpty;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode;
import com.google.devtools.build.skyframe.SkyKey;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.function.Function;
import java.util.function.Predicate;

/**
 * Lays out the selection for upload: numbers the leaves of the node graph, and orders the selected
 * keys. See {@link #computeLeafIdsAndUploadOrder}.
 *
 * <p>An instance is the selection, split for the leaf walk and for upload ordering in a single
 * serial pass.
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
final class UploadLayout {
  /** Nodes for the leaf walk, in selection order. Duplicates are retained; the walk dedups. */
  private final ArrayList<FileOpNode> roots = new ArrayList<>();

  private final ArrayList<ActionLookupKey> unrankedAnalysisKeys = new ArrayList<>();

  /** Parallel to {@link #rankedNodes}. */
  private final ArrayList<ActionLookupKey> rankedAnalysisKeys = new ArrayList<>();

  private final ArrayList<AbstractNestedFileOpNodes> rankedNodes = new ArrayList<>();

  /** Execution keys and {@link ActionLookupSummaryKey}s; see the class comment. */
  private final ArrayList<SkyKey> ownedKeys = new ArrayList<>();

  private UploadLayout() {}

  /**
   * Leaf identifiers, and the order in which to drive uploads, both derived from one walk.
   *
   * @param leafIds the dense leaf numbering
   * @param uploadOrder every selected key that will be uploaded: analysis keys without a nested
   *     node, then analysis keys by the longest-path depth of their node, deepest first, then owned
   *     keys (execution keys and {@link ActionLookupSummaryKey}s). See {@link UploadLayout}.
   */
  public record DenseLeafIdsAndUploadOrder(
      LeafIdAssignment leafIds, ImmutableList<SkyKey> uploadOrder) {}

  /**
   * Numbers every leaf reachable from {@code selection} so that each node's leaf closure is close
   * to a contiguous range, and derives the upload order from the node depths of the same walk.
   *
   * <p>Must run after {@link FileOpNodeMemoizingLookup#materializeNodeGraph}, which is what
   * materializes the node graph, and before any serialization work asks for an ID.
   */
  static DenseLeafIdsAndUploadOrder computeLeafIdsAndUploadOrder(
      ImmutableSet<SkyKey> selection, FileOpNodeMemoizingLookup lookup) {
    UploadLayout partition;
    try (var _ = Profiler.instance().profile("partitionSelection")) {
      partition = partition(selection, lookup::isSkipped, lookup::resolvedNode);
    }
    LeafIdsAndMaxDepth layout = NodeGraphLayout.populateLeafIdsAndDepths(partition.roots);

    ImmutableList<SkyKey> uploadOrder;
    try (var _ = Profiler.instance().profile("uploadOrder")) {
      uploadOrder = partition.uploadOrder(layout.maxDepth());
    }
    return new DenseLeafIdsAndUploadOrder(layout.leafIds(), uploadOrder);
  }

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
  static UploadLayout partition(
      Iterable<? extends SkyKey> selection,
      Predicate<SkyKey> isSkipped,
      Function<ActionLookupKey, FileOpNodeOrEmpty> resolveNode) {
    var partition = new UploadLayout();
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

  /** The leaf numbering, and the greatest depth assigned to any node. */
  private record LeafIdsAndMaxDepth(LeafIdAssignment leafIds, int maxDepth) {}

  /**
   * One depth-first walk of the node graph that lays it out for upload: it numbers the leaves and
   * assigns every nested node its longest-path depth.
   *
   * <p>Numbering leaves in depth-first order puts the leaves of each subtree into a nearly
   * contiguous range, which compresses to a handful of run containers in a {@code RoaringBitmap}.
   *
   * <p>Each node's {@link AbstractNestedFileOpNodes#getLongestPathDepth} field doubles as the
   * visited marker while the walk is running, and {@link #assignLongestPathDepths} overwrites it
   * with the final depth afterwards.
   *
   * <p>Only reachable through {@link #populateLeafIdsAndDepths}, so the walk's scratch state is
   * unreachable once it returns.
   */
  private static final class NodeGraphLayout {
    private final Object2IntOpenHashMap<FileOpNode> analysisIds = newIdMap();
    private final Object2IntOpenHashMap<FileKey> sourceIds = newIdMap();

    /**
     * Every nested node the walk reached, in the order it finished. Consumed by {@link
     * #assignLongestPathDepths}.
     */
    private final ArrayList<AbstractNestedFileOpNodes> postOrder = new ArrayList<>();

    /** Shared by both channels: the two maps land in the same bitmap, so IDs must not collide. */
    private int nextLeafId = 0;

    /**
     * Numbers every leaf reachable from {@code roots} and sets the depth of every nested node
     * reachable from them.
     */
    private static LeafIdsAndMaxDepth populateLeafIdsAndDepths(List<FileOpNode> roots) {
      var layout = new NodeGraphLayout();
      try (var _ = Profiler.instance().profile("leafIdWalk")) {
        for (FileOpNode root : roots) {
          layout.visitRoot(root);
        }
      }
      int maxDepth;
      try (var _ = Profiler.instance().profile("assignLongestPathDepths")) {
        maxDepth = layout.assignLongestPathDepths();
      }
      return new LeafIdsAndMaxDepth(layout.buildLeafIds(), maxDepth);
    }

    private void visitRoot(FileOpNode root) {
      switch (root) {
        case FileKey file -> assignAnalysisId(file);
        case DirectoryListingKey listing -> assignAnalysisId(listing);
        case AbstractNestedFileOpNodes nested -> visitNested(nested);
        case RemoteFileOpNode _ -> {}
      }
    }

    private void visitNested(AbstractNestedFileOpNodes node) {
      // Provisional: assignLongestPathDepths may push the node deeper once all its parents are
      // known.
      if (!node.markVisited()) {
        return;
      }
      for (int i = 0; i < node.analysisDependenciesCount(); i++) {
        switch (node.getAnalysisDependency(i)) {
          case FileKey file -> assignAnalysisId(file);
          case DirectoryListingKey listing -> assignAnalysisId(listing);
          case AbstractNestedFileOpNodes child -> {
            child.incrementInDegree();
            visitNested(child);
          }
          case RemoteFileOpNode _ -> {}
        }
      }
      if (node instanceof NestedFileOpNodesWithSource withSource) {
        assignIdIfAbsent(sourceIds, withSource.source());
      }
      postOrder.add(node);
    }

    /**
     * Assigns every walked node its longest-path depth, and returns the greatest one.
     *
     * <p>In a depth-first post-order, every node finishes before any node that depends on it, so
     * reading {@link #postOrder} backwards visits each node after all its parents. By then its
     * depth is final, and it only has to push the next depth down onto its children. One pass over
     * the nodes and their edges suffices.
     */
    private int assignLongestPathDepths() {
      int maxDepth = AbstractNestedFileOpNodes.TOP_DEPTH;
      for (int i = postOrder.size() - 1; i >= 0; i--) {
        AbstractNestedFileOpNodes node = postOrder.get(i);
        // Since all ancestors have already been traversed, this is the node's final depth.
        int depth = node.getLongestPathDepth();
        maxDepth = max(maxDepth, depth);
        for (int j = 0; j < node.analysisDependenciesCount(); j++) {
          if (node.getAnalysisDependency(j) instanceof AbstractNestedFileOpNodes child) {
            child.placeBelow(depth);
          }
        }
      }
      return maxDepth;
    }

    private void assignAnalysisId(FileOpNode leaf) {
      assignIdIfAbsent(analysisIds, leaf);
    }

    /**
     * Assigns {@code key} the next leaf ID unless it already has one.
     *
     * <p>Uses {@code getInt} followed by {@code put} rather than {@code putIfAbsent(K, int)}, which
     * fastutil 7 lacks. Bazel builds against fastutil 7, where that call resolves to the boxed
     * {@link java.util.Map#putIfAbsent}, which returns null for an absent key.
     */
    private <K> void assignIdIfAbsent(Object2IntOpenHashMap<K> ids, K key) {
      if (ids.getInt(key) == LeafIdAssignment.ABSENT) {
        ids.put(key, nextLeafId++);
      }
    }

    /**
     * A map returning {@link LeafIdAssignment#ABSENT} for missing keys, so that {@code getInt}
     * distinguishes them. fastutil's built-in default, 0, is a real ID.
     */
    private static <K> Object2IntOpenHashMap<K> newIdMap() {
      var map = new Object2IntOpenHashMap<K>();
      map.defaultReturnValue(LeafIdAssignment.ABSENT);
      return map;
    }

    private LeafIdAssignment buildLeafIds() {
      // The populations are final now, so release the slack left by the maps' doubling growth.
      analysisIds.trim();
      sourceIds.trim();
      return LeafIdAssignment.dense(analysisIds, sourceIds);
    }
  }
}
