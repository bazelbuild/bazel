// Copyright 2014 The Bazel Authors. All rights reserved.
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
package com.google.devtools.build.skyframe;

import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.lib.supplier.InterruptibleSupplier;
import com.google.devtools.build.lib.supplier.MemoizingInterruptibleSupplier;
import com.google.errorprone.annotations.CanIgnoreReturnValue;
import java.util.List;
import javax.annotation.Nullable;

/**
 * Interface between a single version of the graph and the evaluator. Supports mutation of that
 * single version of the graph.
 *
 * <p>Certain graph implementations can throw {@link InterruptedException} when trying to retrieve
 * node entries. Such exceptions should not be caught locally -- they should be allowed to propagate
 * up.
 *
 * <p>This class is not intended for direct use, and is only exposed as public for use in evaluation
 * implementations outside of this package.
 */
@ThreadSafe
public interface ProcessableGraph extends QueryableGraph {

  /** Remove the value with given name from the graph. */
  void remove(SkyKey key);

  /**
   * Like {@link QueryableGraph#getBatch}, except creates a new node for each key not already
   * present in the graph.
   *
   * <p>By the time this method returns, nodes are guaranteed to have been created if necessary for
   * each requested key. It is not necessary to call {@link NodeBatch#get} to trigger node creation.
   *
   * <p>Calling {@link NodeBatch#get} on the returned batch will never return {@code null} for any
   * key in {@code keys}. Even if there is an intervening call to {@link #remove}, the call to
   * {@link NodeBatch#get} will re-create a {@link NodeEntry} if necessary.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  @CanIgnoreReturnValue
  NodeBatch createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<? extends SkyKey> keys)
      throws InterruptedException;

  /**
   * Like {@link QueryableGraph#getBatchAsync}, except creates a new node for each key not already
   * present in the graph. Thus, calling {@link NodeBatch#get} on the returned batch will never
   * return {@code null} for any of the requested {@code keys}.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  @CanIgnoreReturnValue
  default InterruptibleSupplier<NodeBatch> createIfAbsentBatchAsync(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
    return MemoizingInterruptibleSupplier.of(() -> createIfAbsentBatch(requestor, reason, keys));
  }

  /**
   * Optional optimization: graph may use internal knowledge to filter out keys in {@code deps} that
   * have not been recomputed since the last computation of {@code parent}. When determining if
   * {@code parent} needs to be re-evaluated, this may be used to avoid unnecessary graph accesses.
   *
   * <p>If this graph partakes in the optional optimization, returns deps that may have new values
   * since the node of {@code parent} was last computed, and therefore which may force re-evaluation
   * of the node of {@code parent}. Otherwise, returns {@link DepsReport#NO_INFORMATION}.
   *
   * @param parent the key in {@link NodeEntry.LifecycleState#CHECK_DEPENDENCIES}
   * @param deps the {@linkplain NodeEntry#getNextDirtyDirectDeps next dirty dep group} of {@code
   *     parent}; only called when all previous dep groups were clean, so it is known that {@code
   *     deps} are still dependencies of {@code parent} on the incremental build
   */
  DepsReport analyzeDepsDoneness(SkyKey parent, List<SkyKey> deps) throws InterruptedException;
}
