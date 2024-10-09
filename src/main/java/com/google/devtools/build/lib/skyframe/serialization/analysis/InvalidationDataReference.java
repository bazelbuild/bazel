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

import static com.google.common.base.Preconditions.checkState;

import com.google.common.util.concurrent.AbstractFuture;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.devtools.build.lib.vfs.RootedPath;
import javax.annotation.Nullable;

/**
 * Future representing transitive write status for file or directory invalidation data.
 *
 * <p>The client should perform double-checked locking to populate some of the metadata by testing
 * {@link #getCacheKey} for nullness. If it is null outside of the lock, and null again inside the
 * lock, the caller must call either one of the {@code populate} methods depending on the subclass.
 */
sealed class InvalidationDataReference extends AbstractFuture<Void>
    permits InvalidationDataReference.FileInvalidationDataReference,
        InvalidationDataReference.DirectoryInvalidationDataReference {
  private volatile String cacheKey;

  @Nullable // null when uninitialized
  String getCacheKey() {
    return cacheKey;
  }

  void setWriteStatus(ListenableFuture<Void> writeStatus) {
    checkState(setFuture(writeStatus), "already set %s", this);
  }

  void setUnexpectedlyUnsetError() {
    checkState(
        setException(
            new IllegalStateException(
                "future was unexpectedly unset, look for unchecked exceptions in"
                    + " FileDependencySerializer")));
  }

  void setCacheKeyForSubclasses(String cacheKey) {
    this.cacheKey = cacheKey;
  }

  static final class FileInvalidationDataReference extends InvalidationDataReference {
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

  static final class DirectoryInvalidationDataReference extends InvalidationDataReference {
    void populate(String cacheKey) {
      setCacheKeyForSubclasses(cacheKey);
    }
  }
}
