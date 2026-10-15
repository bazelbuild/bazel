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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.FileOpNode;
import it.unimi.dsi.fastutil.objects.Object2IntOpenHashMap;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import javax.annotation.Nullable;

/**
 * A precomputed, dense assignment of integer IDs to leaves of the {@link FileOpNode} graph.
 *
 * <p>Every leaf ID becomes a bit position in a {@link org.roaringbitmap.RoaringBitmap} recording
 * the transitive leaf closure of a node. Roaring's run-length encoding only pays off when the bits
 * set in a single bitmap are contiguous, so the IDs are assigned by a depth-first walk of the node
 * graph (see {@link FileOpNodeMemoizingLookup#assignDenseLeafIds}): the leaves beneath any given
 * node are numbered consecutively at the point that node is first reached, which makes that node's
 * closure a small number of runs rather than a scatter of isolated bits.
 *
 * <p>IDs cannot be stored on the leaves themselves. {@link FileKey} and {@link DirectoryListingKey}
 * are globally interned and outlive the build, so a mutable per-build ID field on them would be
 * shared across invocations. Hence the side maps here.
 *
 * <p>Analysis and source dependencies are kept in separate maps so that a file registered as an
 * analysis dependency does not collide with or prune a source dependency on the same file. Both
 * draw from a single ID sequence, because both land in the same bitmap.
 *
 * <h2>Leaves in dense vs. lazy mode</h2>
 *
 * <p>In dense mode, every leaf queried was reached by the node graph walk (see {@link
 * FileOpNodeMemoizingLookup#assignDenseLeafIds}). Non-leaf dependencies such as ancestor
 * directories, symlink parents, and unreferenced source files never query for an ID. A miss in
 * dense mode is therefore a bug, and dense mode fails fast with {@link IllegalStateException}
 * without allocating fallback maps.
 *
 * <p>Only {@link #lazy()} keeps concurrent fallback maps, which numbers leaves on demand for
 * callers that serialize without a materialized node graph.
 */
final class LeafIdAssignment {
  /** Returned by the dense maps for a leaf they do not hold. Real IDs are non-negative. */
  private static final int ABSENT = -1;

  /**
   * Dense IDs, keyed on leaf identity.
   *
   * <p>Primitive-valued to avoid boxing: at this population the {@link Integer} objects alone would
   * cost more than the rest of the map. Neither map is mutated after construction, so concurrent
   * reads during serialization are safe, and the final fields here publish their contents.
   */
  private final Object2IntOpenHashMap<FileOpNode> analysisIds;

  private final Object2IntOpenHashMap<FileKey> sourceIds;

  /**
   * IDs for leaves that were not reached by the graph walk, used only in lazy mode.
   *
   * <p>Null in dense mode, where any unnumbered leaf is an unexpected omission.
   */
  @Nullable private final ConcurrentHashMap<FileOpNode, Integer> fallbackAnalysisIds;

  @Nullable private final ConcurrentHashMap<FileKey, Integer> fallbackSourceIds;
  @Nullable private final AtomicInteger nextFallbackId;

  /**
   * An assignment with no dense range, numbering every leaf on demand.
   *
   * <p>Used by callers that serialize as they compute, where there is no completed node graph to
   * walk, and by tests that exercise the serializer without building one.
   */
  static LeafIdAssignment lazy() {
    return new LeafIdAssignment();
  }

  /**
   * An assignment holding only the given dense IDs, failing fast on any other leaf.
   *
   * <p>Takes ownership of the two maps. The caller must not retain or mutate them afterwards.
   */
  static LeafIdAssignment dense(
      Object2IntOpenHashMap<FileOpNode> analysisIds, Object2IntOpenHashMap<FileKey> sourceIds) {
    return new LeafIdAssignment(analysisIds, sourceIds);
  }

  private LeafIdAssignment() {
    this.analysisIds = new Object2IntOpenHashMap<>();
    this.sourceIds = new Object2IntOpenHashMap<>();
    this.analysisIds.defaultReturnValue(ABSENT);
    this.sourceIds.defaultReturnValue(ABSENT);
    this.fallbackAnalysisIds = new ConcurrentHashMap<>();
    this.fallbackSourceIds = new ConcurrentHashMap<>();
    this.nextFallbackId = new AtomicInteger(0);
  }

  private LeafIdAssignment(
      Object2IntOpenHashMap<FileOpNode> analysisIds, Object2IntOpenHashMap<FileKey> sourceIds) {
    analysisIds.defaultReturnValue(ABSENT);
    sourceIds.defaultReturnValue(ABSENT);
    this.analysisIds = analysisIds;
    this.sourceIds = sourceIds;
    this.fallbackAnalysisIds = null;
    this.fallbackSourceIds = null;
    this.nextFallbackId = null;
  }

  /** The number of leaves that received a dense ID. */
  int denseIdCount() {
    return analysisIds.size() + sourceIds.size();
  }

  /** The number of leaves that were assigned an ID on demand in lazy mode. */
  int fallbackIdCount() {
    return fallbackAnalysisIds != null ? fallbackAnalysisIds.size() + fallbackSourceIds.size() : 0;
  }

  /** The ID of an analysis dependency on a file. */
  int getAnalysisId(FileKey key) {
    return lookupAnalysisId(key);
  }

  /**
   * The ID of an analysis dependency on a directory listing.
   *
   * <p>There is no source-channel counterpart: a directory listing is never a source dependency.
   */
  int getAnalysisId(DirectoryListingKey key) {
    return lookupAnalysisId(key);
  }

  /**
   * The ID of a source dependency on a file.
   *
   * <p>Deliberately distinct from {@link #getAnalysisId(FileKey)}. The same file registered through
   * both channels gets two IDs, so that an analysis dependency on it neither collides with nor
   * prunes a source dependency on it.
   */
  int getSourceId(FileKey key) {
    int id = sourceIds.getInt(key);
    if (id != ABSENT) {
      return id;
    }
    checkState(fallbackSourceIds != null, "Missing dense leaf ID for source file %s", key);
    return fallbackSourceIds.computeIfAbsent(
        key, _ -> checkNotNull(nextFallbackId).getAndIncrement());
  }

  private int lookupAnalysisId(FileOpNode key) {
    int id = analysisIds.getInt(key);
    if (id != ABSENT) {
      return id;
    }
    checkState(fallbackAnalysisIds != null, "Missing dense leaf ID for %s", key);
    return fallbackAnalysisIds.computeIfAbsent(
        key, _ -> checkNotNull(nextFallbackId).getAndIncrement());
  }
}
