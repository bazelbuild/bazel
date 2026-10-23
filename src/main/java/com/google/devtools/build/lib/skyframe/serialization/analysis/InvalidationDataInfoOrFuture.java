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
package com.google.devtools.build.lib.skyframe.serialization.analysis;

import com.google.common.annotations.VisibleForTesting;
import com.google.devtools.build.lib.concurrent.SettableFutureKeyedValue;
import com.google.devtools.build.lib.skyframe.AbstractNestedFileOpNodes;
import com.google.devtools.build.lib.skyframe.DirectoryListingKey;
import com.google.devtools.build.lib.skyframe.FileKey;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.skyframe.serialization.WriteStatus;
import com.google.devtools.build.lib.vfs.RootedPath;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.VarHandle;
import java.util.function.BiConsumer;
import javax.annotation.Nullable;
import org.roaringbitmap.RoaringBitmap;

/**
 * Information about remotely stored invalidation data.
 *
 * <p>There are 3 distinct type families, associated with files, directory listings and nodes
 * (nested sets of files and directory listings).
 *
 * <p>Each family has 3 types, a constant type (no persisted data), information about stored
 * invalidation data or a future. Information about stored invalidation data always includes a cache
 * key and a write status.
 *
 * <p>In the case of {@link FileInvalidationData}, a Max Transitive Source Version (MTSV) and fully
 * resolved path is also included. These fields are used for ancestor resolution.
 */
@SuppressWarnings("InterfaceWithOnlyStatics") // sealed hierarchy root
sealed interface InvalidationDataInfoOrFuture
    permits InvalidationDataInfoOrFuture.InvalidationDataInfo,
        InvalidationDataInfoOrFuture.FileDataInfoOrFuture,
        InvalidationDataInfoOrFuture.ListingDataInfoOrFuture,
        InvalidationDataInfoOrFuture.NodeDataInfoOrFuture {

  /** Non-future, immediate value sub-types of {@link InvalidationDataInfoOrFuture}. */
  sealed interface InvalidationDataInfo extends InvalidationDataInfoOrFuture
      permits FileDataInfo, ListingDataInfo, NodeDataInfo {}

  /** Base implementation of a {@link InvalidationDataInfoOrFuture} value. */
  abstract static sealed class BaseInvalidationDataInfo<T> {
    private final T cacheKey;
    private final WriteStatus writeStatus;

    BaseInvalidationDataInfo(T cacheKey, WriteStatus writeStatus) {
      this.cacheKey = cacheKey;
      this.writeStatus = writeStatus;
    }

    /** Key for {@link com.google.devtools.build.lib.serialization.FingerprintValueService}. */
    final T cacheKey() {
      return cacheKey;
    }

    /** Transitively inclusive status of writing this data to the cache. */
    final WriteStatus writeStatus() {
      return writeStatus;
    }
  }

  sealed interface FileDataInfoOrFuture extends InvalidationDataInfoOrFuture
      permits FileDataInfo, FutureFileDataInfo {}

  sealed interface FileDataInfo extends FileDataInfoOrFuture, InvalidationDataInfo
      permits ConstantFileData, FileInvalidationDataInfo {}

  /** The file doesn't change and isn't associated with any invalidation data. */
  enum ConstantFileData implements FileDataInfo {
    CONSTANT_FILE;
  }

  /** Information about transitive upload of invalidation data for a certain {@link FileKey}. */
  static final class FileInvalidationDataInfo extends BaseInvalidationDataInfo<String>
      implements FileDataInfo {
    private final boolean exists;
    private final long mtsv;
    private final RootedPath realPath;

    FileInvalidationDataInfo(
        String cacheKey, WriteStatus writeStatus, boolean exists, long mtsv, RootedPath realPath) {
      super(cacheKey, writeStatus);
      this.exists = exists;
      this.mtsv = mtsv;
      this.realPath = realPath;
    }

    /** True if the file exists. */
    boolean exists() {
      return exists;
    }

    /**
     * The MTSV.
     *
     * <p>Used by dependents to create parent references. This information is already incorporated
     * into the {@link #cacheKey} value.
     */
    long mtsv() {
      return mtsv;
    }

    /**
     * The resolved real path.
     *
     * <p>Used for symlink resolution.
     */
    RootedPath realPath() {
      return realPath;
    }
  }

  static final class FutureFileDataInfo
      extends SettableFutureKeyedValue<FutureFileDataInfo, FileKey, FileDataInfo>
      implements FileDataInfoOrFuture {
    FutureFileDataInfo(FileKey key, BiConsumer<FileKey, FileDataInfo> consumer) {
      super(key, consumer);
    }
  }

  sealed interface ListingDataInfoOrFuture extends InvalidationDataInfoOrFuture
      permits ListingDataInfo, FutureListingDataInfo {}

  sealed interface ListingDataInfo extends ListingDataInfoOrFuture, InvalidationDataInfo
      permits ConstantListingData, ListingInvalidationDataInfo {}

  /** This listing doesn't change and isn't associated with invalidation data. */
  enum ConstantListingData implements ListingDataInfo {
    CONSTANT_LISTING;
  }

  /**
   * Information about transitive upload of invalidation data for a certain {@link
   * DirectoryListingKey}.
   */
  static final class ListingInvalidationDataInfo extends BaseInvalidationDataInfo<String>
      implements ListingDataInfo {
    ListingInvalidationDataInfo(String cacheKey, WriteStatus writeStatus) {
      super(cacheKey, writeStatus);
    }
  }

  static final class FutureListingDataInfo
      extends SettableFutureKeyedValue<FutureListingDataInfo, DirectoryListingKey, ListingDataInfo>
      implements ListingDataInfoOrFuture {
    FutureListingDataInfo(
        DirectoryListingKey key, BiConsumer<DirectoryListingKey, ListingDataInfo> consumer) {
      super(key, consumer);
    }
  }

  sealed interface NodeDataInfoOrFuture extends InvalidationDataInfoOrFuture
      permits NodeDataInfo, FutureNodeDataInfo {}

  sealed interface NodeDataInfo extends NodeDataInfoOrFuture, InvalidationDataInfo
      permits ConstantNodeData, NodeInvalidationDataInfo {
    /**
     * The number of distinct leaves in this node's locally known transitive closure, or {@link
     * Integer#MAX_VALUE} if {@link #hasOpaqueDependencies}.
     *
     * <p>Orders a parent's children for transitive reduction, largest first, so that a child is
     * only tested against siblings at least as large. Opaque nodes sort first because their true
     * closure is unknown.
     */
    int cardinality();

    /**
     * Whether this node's transitive closure includes a {@link
     * com.google.devtools.build.lib.skyframe.FileOpNodeOrFuture.RemoteFileOpNode}: a node
     * downloaded from the remote cache, whose dependencies are not known locally. That is either
     * this node itself or one of its descendants.
     *
     * <p>No local bitmap can prove that an opaque node is covered by its siblings, so transitive
     * reduction always retains it.
     *
     * <p>This makes the serialized data non-deterministic, but still correct, when a build both
     * downloads and uploads. Whether a given subgraph is remote or local depends on what that build
     * happened to download, so the same node can be serialized with a different set of retained
     * edges, and therefore under a different key, from one build to the next. Every representation
     * still describes the same set of file dependencies, because an opaque child is only ever
     * retained, never dropped, so invalidation is unaffected. The cost is less sharing of cache
     * entries between builds.
     *
     * <p>TODO(b/575653561): Make serialization deterministic in the presence of opaque
     * dependencies.
     */
    boolean hasOpaqueDependencies();
  }

  enum ConstantNodeData implements NodeDataInfo {
    CONSTANT_NODE;

    @Override
    public int cardinality() {
      return 0;
    }

    @Override
    public boolean hasOpaqueDependencies() {
      return false;
    }
  }

  /** Information about remotely persisted {@link AbstractNestedFileOpNodes}. */
  static final class NodeInvalidationDataInfo extends BaseInvalidationDataInfo<PackedFingerprint>
      implements NodeDataInfo {
    private static final VarHandle REMAINING_REFERENCES_HANDLE;

    static {
      try {
        REMAINING_REFERENCES_HANDLE =
            MethodHandles.lookup()
                .findVarHandle(NodeInvalidationDataInfo.class, "remainingReferences", int.class);
      } catch (ReflectiveOperationException e) {
        throw new ExceptionInInitializerError(e);
      }
    }

    @Nullable private RoaringBitmap bitmap;
    private final boolean hasOpaqueDependencies;

    /**
     * Cached because the bitmap is run-optimized, so {@link RoaringBitmap#getCardinality} costs
     * O(containers + runs), and {@code FileDependencySerializer.NODE_DEPENDENCY_COMPARATOR} reads
     * it O(k log k) times per parent with k children.
     */
    private final int cardinality;

    private final boolean pinned;
    private volatile int remainingReferences;

    NodeInvalidationDataInfo(
        PackedFingerprint key,
        WriteStatus writeStatus,
        RoaringBitmap bitmap,
        boolean hasOpaqueDependencies,
        int initialReferences) {
      this(key, writeStatus, bitmap, hasOpaqueDependencies, initialReferences, /* pinned= */ false);
    }

    private NodeInvalidationDataInfo(
        PackedFingerprint key,
        WriteStatus writeStatus,
        RoaringBitmap bitmap,
        boolean hasOpaqueDependencies,
        int initialReferences,
        boolean pinned) {
      super(key, writeStatus);
      this.bitmap = bitmap;
      this.hasOpaqueDependencies = hasOpaqueDependencies;
      // Opaque nodes can never be pruned (see hasOpaqueDependencies), and siblings are visited in
      // descending cardinality. Ranking opaque nodes first merges their bits into the union early,
      // which gives the most opportunity to prune the siblings that follow.
      this.cardinality = hasOpaqueDependencies ? Integer.MAX_VALUE : bitmap.getCardinality();
      this.pinned = pinned;
      this.remainingReferences = initialReferences;
    }

    /**
     * Creates an info that ignores reference counting and never drops its bitmap.
     *
     * <p>For nodes whose incoming edges are not counted, where any initial count could reach zero
     * while parents still need the bitmap.
     */
    static NodeInvalidationDataInfo pinned(
        PackedFingerprint key,
        WriteStatus writeStatus,
        RoaringBitmap bitmap,
        boolean hasOpaqueDependencies) {
      return new NodeInvalidationDataInfo(
          key,
          writeStatus,
          bitmap,
          hasOpaqueDependencies,
          /* initialReferences= */ 0,
          /* pinned= */ true);
    }

    public void decrementReference() {
      if (pinned) {
        return;
      }
      if ((int) REMAINING_REFERENCES_HANDLE.getAndAdd(this, -1) == 1) {
        this.bitmap = null;
      }
    }

    public void addReferences(int delta) {
      if (pinned) {
        return;
      }
      REMAINING_REFERENCES_HANDLE.getAndAdd(this, delta);
    }

    /**
     * Merges this node's leaf closure into {@code union}, the closure of the siblings retained so
     * far.
     *
     * @return true if this node must be retained as a dependency: it is opaque, or it contributed
     *     at least one bit not already in {@code union}. False means its closure was already
     *     covered, so the edge to it is redundant and can be dropped.
     */
    public boolean mergeInto(RoaringBitmap union) {
      if (bitmap == null) {
        throw new IllegalStateException("Attempted to merge from dropped bitmap for " + cacheKey());
      }
      if (hasOpaqueDependencies) {
        union.or(bitmap);
        return true;
      }
      // Checks whether the union grew instead of using union.contains(bitmap), which can wrongly
      // report a subset (b/565949795).
      long before = union.getLongCardinality();
      union.or(bitmap);
      return union.getLongCardinality() != before;
    }

    @Override
    public int cardinality() {
      return cardinality;
    }

    @Override
    public boolean hasOpaqueDependencies() {
      return hasOpaqueDependencies;
    }

    @VisibleForTesting
    boolean isBitmapDroppedForTesting() {
      return bitmap == null;
    }

    @VisibleForTesting
    int getRemainingReferencesForTesting() {
      return remainingReferences;
    }
  }

  static final class FutureNodeDataInfo
      extends SettableFutureKeyedValue<FutureNodeDataInfo, AbstractNestedFileOpNodes, NodeDataInfo>
      implements NodeDataInfoOrFuture {
    FutureNodeDataInfo(AbstractNestedFileOpNodes key) {
      super(key, FutureNodeDataInfo::setNodeDataInfo);
    }

    private static void setNodeDataInfo(AbstractNestedFileOpNodes key, NodeDataInfo value) {
      key.setSerializationScratch(value);
    }
  }
}
