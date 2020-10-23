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
import java.util.Collection;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * Interface between a single version of the graph and the evaluator. Supports mutation of that
 * single version of the graph.
 *
 * <p>Certain graph implementations can throw {@link InterruptedException} when trying to retrieve
 * node entries. Such exceptions should not be caught locally -- they should be allowed to propagate
 * up.
 */
@ThreadSafe
interface EvaluableGraph extends QueryableGraph, DeletableGraph {
  /**
   * Like {@link QueryableGraph#getBatch}, except it creates a new node for each key not already
   * present in the graph. Thus, the returned map will have an entry for each key in {@code keys}.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  Map<SkyKey, ? extends NodeEntry> createIfAbsentBatch(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) throws InterruptedException;

  /**
   * Like {@link QueryableGraph#getBatchAsync}, except it creates a new node for each key not
   * already present in the graph. Thus, the returned map will have an entry for each key in {@code
   * keys}.
   *
   * @param requestor if non-{@code null}, the node on behalf of which the given {@code keys} are
   *     being requested.
   * @param reason the reason the nodes are being requested.
   */
  @CanIgnoreReturnValue
  default InterruptibleSupplier<Map<SkyKey, ? extends NodeEntry>> createIfAbsentBatchAsync(
      @Nullable SkyKey requestor, Reason reason, Iterable<SkyKey> keys) {
    return MemoizingInterruptibleSupplier.of(() -> createIfAbsentBatch(requestor, reason, keys));
  }

  /**
   * Optional optimization: graph may use internal knowledge to filter out keys in {@code deps} that
   * have not been recomputed since the last computation of {@code parent}. When determining if
   * {@code parent} needs to be re-evaluated, this may be used to avoid unnecessary graph accesses.
   *
   * <p>Returns deps that may have new values since the node of {@code parent} was last computed,
   * and therefore which may force re-evaluation of the node of {@code parent}.
   */
  DepsReport analyzeDepsDoneness(SkyKey parent, Collection<SkyKey> deps)
      throws InterruptedException;

}
