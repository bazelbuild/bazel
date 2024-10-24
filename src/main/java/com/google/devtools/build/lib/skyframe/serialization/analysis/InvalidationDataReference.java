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

import static com.google.common.base.Preconditions.checkNotNull;
import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.skyframe.serialization.PackedFingerprint;
import com.google.devtools.build.lib.vfs.RootedPath;
import javax.annotation.Nullable;

/**
 * Future representing the transitive write status of invalidation data.
 *
 * <p>In the case of {@link FileInvalidationDataReference} and {@link
 * ListingInvalidationDataReference}, the client should perform double-checked locking to populate
 * the metadata by testing {@link #getCacheKey} for nullness. If it is null outside of the lock, and
 * null again inside the lock, the caller must call one of the {@code populate} methods depending on
 * the subclass.
 *
 * <p>For {@link NodeInvalidationDataReference}, the metadata is assigned at construction.
 */
sealed class InvalidationDataReference<T> extends AbstractFuture<Void>
    permits InvalidationDataReference.FileInvalidationDataReference,
        InvalidationDataReference.ListingInvalidationDataReference,
        InvalidationDataReference.NodeInvalidationDataReference {
  private volatile T cacheKey;

  @Nullable // null when uninitialized
  final T getCacheKey() {
    return cacheKey;
  }

  final void setWriteStatus(ListenableFuture<Void> writeStatus) {
    checkState(setFuture(writeStatus), "already set %s", this);
  }

  final void setUnexpectedlyUnsetError() {
    checkState(
        setException(
            new IllegalStateException(
                "future was unexpectedly unset, look for unchecked exceptions in"
                    + " FileDependencySerializer")));
  }

  final void setCacheKeyForSubclasses(T cacheKey) {
    this.cacheKey = cacheKey;
  }

  /** Future transitive write status for {@link com.google.devtools.build.lib.skyframe.FileKey}. */
  static final class FileInvalidationDataReference extends InvalidationDataReference<String> {
    private long mtsv;
    private RootedPath realPath;

    void populate(long mtsv, RootedPath realPath, String cacheKey) {
      // https://docs.oracle.com/javase/specs/jls/se21/html/jls-17.html#jls-17.4.5
      //
      // "If x and y are actions of the same thread and x comes before y in program order, then
      // hb(x, y)." (where "hb" stands for happens-before)
      //
      // `mtsv` and `realPath` are set prior to `cacheKey` so any thread that observes the volatile
      // `cacheKey` write also observe the plain `mtsv` and `realPath` writes.
      this.mtsv = mtsv;
      this.realPath = realPath;
      setCacheKeyForSubclasses(cacheKey);
    }

    /**
     * Returns the MTSV.
     *
     * <p>Value undefined before {@link #populate} completes.
     */
    long getMtsv() {
      return mtsv;
    }

    /**
     * Returns the resolved real path.
     *
     * <p>Value undefined before {@link #populate} completes.
     */
    RootedPath getRealPath() {
      return realPath;
    }
  }

  /**
   * Future transitive write status for {@link
   * com.google.devtools.build.lib.skyframe.DirectoryListingKey}.
   */
  static final class ListingInvalidationDataReference extends InvalidationDataReference<String> {
    void populate(String cacheKey) {
      setCacheKeyForSubclasses(cacheKey);
    }
  }

  /**
   * Future transitive write status of {@link
   * com.google.devtools.build.lib.skyframe.NestedFileSystemOperationNodes}.
   *
   * <p>{@link #getCacheKey} is null only for {@link #EMPTY}.
   */
  static final class NodeInvalidationDataReference
      extends InvalidationDataReference<PackedFingerprint> {
    /* In contrast to the above two implementations, there's no distinct populate phase because
     * this class does not need to be constructed inside a computeIfAbsent callback.
     */

    /**
     * A no-data sentinel value.
     *
     * <p>Occurs if none of the transitive node inputs are relevant to invalidation.
     */
    static final NodeInvalidationDataReference EMPTY = new NodeInvalidationDataReference(null);

    static NodeInvalidationDataReference create(PackedFingerprint key) {
      return new NodeInvalidationDataReference(checkNotNull(key));
    }

    private NodeInvalidationDataReference(PackedFingerprint key) {
      setCacheKeyForSubclasses(key);
    }
  }
}
