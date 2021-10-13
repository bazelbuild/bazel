// Copyright 2021 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkArgument;
import static com.google.common.util.concurrent.Futures.immediateVoidFuture;
import static com.google.common.util.concurrent.MoreExecutors.directExecutor;

import com.github.benmanes.caffeine.cache.Cache;
import com.github.benmanes.caffeine.cache.Caffeine;
import com.google.common.util.concurrent.FutureCallback;
import com.google.common.util.concurrent.Futures;
import com.google.common.util.concurrent.ListenableFuture;
import com.google.common.util.concurrent.SettableFuture;
import com.google.devtools.build.lib.bugreport.BugReporter;
import com.google.devtools.build.lib.collect.nestedset.NestedSetStore.FingerprintComputationResult;
import com.google.devtools.build.lib.skyframe.serialization.SerializationConstants;
import com.google.protobuf.ByteString;
import javax.annotation.Nullable;

/**
 * A bidirectional, in-memory, weak cache for fingerprint ⟺ {@link NestedSet} associations.
 *
 * <p>For use by {@link NestedSetStore} to minimize work during {@link NestedSet} (de)serialization.
 */
class NestedSetSerializationCache {

  /**
   * Fingerprint to array cache.
   *
   * <p>The values in this cache are always {@code Object[]} or {@code ListenableFuture<Object[]>}.
   * We avoid a common wrapper object both for memory efficiency and because our cache eviction
   * policy is based on value GC, and wrapper objects would defeat that.
   *
   * <p>While a fetch for the contents is outstanding, the value in the cache will be a {@link
   * ListenableFuture}. When it is resolved, it is replaced with the unwrapped {@code Object[]}.
   * This is done because if the array is a transitive member, its future may be GC'd, and we want
   * entries to stay in this cache while the contents are still live.
   */
  private final Cache<ByteString, Object> fingerprintToContents =
      Caffeine.newBuilder()
          .initialCapacity(SerializationConstants.DESERIALIZATION_POOL_SIZE)
          .weakValues()
          .build();

  /** {@code Object[]} contents to fingerprint. Maintained for fast fingerprinting. */
  private final Cache<Object[], FingerprintComputationResult> contentsToFingerprint =
      Caffeine.newBuilder()
          .initialCapacity(SerializationConstants.DESERIALIZATION_POOL_SIZE)
          .weakKeys()
          .build();

  private final BugReporter bugReporter;

  NestedSetSerializationCache(BugReporter bugReporter) {
    this.bugReporter = bugReporter;
  }

  /**
   * Returns contents (an {@code Object[]} or a {@code ListenableFuture<Object[]>}) for the {@link
   * NestedSet} associated with the given fingerprint if there was already one. Otherwise associates
   * {@code future} with {@code fingerprint} and returns {@code null}.
   *
   * <p>Upon a {@code null} return, the caller should ensure that the given future is eventually set
   * with the fetched contents.
   *
   * <p>Upon a non-{@code null} return, the caller should discard the given future in favor of the
   * returned contents, blocking for them if the return value is itself a future.
   *
   * @param fingerprint the fingerprint of the desired {@link NestedSet} contents
   * @param future a freshly created {@link SettableFuture}
   */
  @Nullable
  Object putIfAbsent(ByteString fingerprint, SettableFuture<Object[]> future) {
    checkArgument(!future.isDone(), "Must pass a fresh future: %s", future);
    Object existing = fingerprintToContents.asMap().putIfAbsent(fingerprint, future);
    if (existing != null) {
      return existing;
    }
    // This is the first request of this fingerprint.
    unwrapWhenDone(fingerprint, future);
    return null;
  }

  /**
   * Registers a {@link FutureCallback} that associates the provided {@code fingerprint} and the
   * contents of the future, when it completes.
   *
   * <p>There may be a race between this call and calls to {@link #put}. Those races are benign,
   * since the fingerprint should be the same regardless. We may pessimistically end up having a
   * future to wait on for serialization that isn't actually necessary, but that isn't a big
   * concern.
   */
  private void unwrapWhenDone(ByteString fingerprint, ListenableFuture<Object[]> futureContents) {
    Futures.addCallback(
        futureContents,
        new FutureCallback<Object[]>() {
          @Override
          public void onSuccess(Object[] contents) {
            // Replace the cache entry with the unwrapped contents, since the Future may be GC'd.
            fingerprintToContents.put(fingerprint, contents);
            // There may already be an entry here, but it's better to put a fingerprint result with
            // an immediate future, since then later readers won't need to block unnecessarily. It
            // would be nice to check the old value, but Cache#put doesn't provide it to us.
            contentsToFingerprint.put(
                contents, FingerprintComputationResult.create(fingerprint, immediateVoidFuture()));
          }

          @Override
          public void onFailure(Throwable t) {
            // Failure to fetch the NestedSet contents is unexpected, but the failed future can be
            // stored as the NestedSet children. This way the exception is only propagated if the
            // NestedSet is consumed (unrolled).
            bugReporter.sendBugReport(t);
          }
        },
        directExecutor());
  }

  /**
   * Retrieves the fingerprint associated with the given {@link NestedSet} contents, or {@code null}
   * if the given contents are not known.
   */
  @Nullable
  FingerprintComputationResult fingerprintForContents(Object[] contents) {
    return contentsToFingerprint.getIfPresent(contents);
  }

  // TODO(janakr): Currently, racing threads can overwrite each other's
  // fingerprintComputationResult, leading to confusion and potential performance drag. Fix this.
  void put(FingerprintComputationResult result, Object[] contents) {
    contentsToFingerprint.put(contents, result);
    fingerprintToContents.put(result.fingerprint(), contents);
  }
}
