// Copyright 2015 The Bazel Authors. All rights reserved.
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

import com.google.common.base.Preconditions;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadSafe;
import com.google.devtools.build.skyframe.NodeEntry.DirtyState;
import javax.annotation.Nullable;

/**
 * A node in the graph without the means to access its value. All operations on this class are
 * thread-safe (note, however, the warning on the return value of {@link #markDirty}).
 *
 * <p>This interface is public only for the benefit of alternative graph implementations outside of
 * the package.
 */
public interface ThinNodeEntry {

  /** Returns whether the entry has been built and is finished evaluating. */
  @ThreadSafe
  boolean isDone();

  /**
   * Returns true if the entry is new or marked as dirty. This includes the case where its deps are
   * still being checked for up-to-dateness.
   */
  @ThreadSafe
  boolean isDirty();

  /**
   * Returns true if the entry is marked changed, meaning that it must be re-evaluated even if its
   * dependencies' values have not changed.
   */
  @ThreadSafe
  boolean isChanged();

  /** Ways that a node may be dirtied. */
  enum DirtyType {
    /**
     * A node P dirtied with DIRTY is re-evaluated during the evaluation phase if it's requested and
     * directly depends on some node C whose value changed since the last evaluation of P. If it's
     * requested and there is no such node C, P is marked clean.
     */
    DIRTY(DirtyState.CHECK_DEPENDENCIES),

    /**
     * A node dirtied with CHANGE is re-evaluated during the evaluation phase if it's requested
     * (regardless of the state of its dependencies). Such a node is expected to evaluate to the
     * same value if evaluated at the same graph version.
     */
    CHANGE(DirtyState.NEEDS_REBUILDING),

    /**
     * A node dirtied with FORCE_REBUILD behaves like a {@link #CHANGE}d node, except that it may
     * evaluate to a different value even if evaluated at the same graph version.
     */
    FORCE_REBUILD(DirtyState.NEEDS_FORCED_REBUILDING);

    private final DirtyState initialDirtyState;

    DirtyType(DirtyState initialDirtyState) {
      this.initialDirtyState = initialDirtyState;
    }

    DirtyState getInitialDirtyState() {
      return initialDirtyState;
    }
  }

  /**
   * Marks this node dirty as specified by the provided {@link DirtyType}.
   *
   * <p>{@code markDirty(DirtyType.DIRTY)} may only be called on a node P for which {@code
   * P.isDone() || P.isChanged()} (the latter is permitted but has no effect). Similarly, {@code
   * markDirty(DirtyType.CHANGE)} may only be called on a node P for which {@code P.isDone() ||
   * !P.isChanged()}. Otherwise, this will throw {@link IllegalStateException}.
   *
   * <p>{@code markDirty(DirtyType.FORCE_REBUILD)} may be called multiple times; only the first has
   * any effect.
   *
   * @return if the node was done, a {@link MarkedDirtyResult} which may include the node's reverse
   *     deps; otherwise {@code null}
   */
  @Nullable
  @ThreadSafe
  MarkedDirtyResult markDirty(DirtyType dirtyType) throws InterruptedException;

  /**
   * Returned by {@link #markDirty} if that call changed the node from done to dirty. Contains an
   * iterable of the node's reverse deps for efficiency, because an important use case for {@link
   * #markDirty} is during invalidation, and the invalidator must immediately afterwards schedule
   * the invalidation of a node's reverse deps if the invalidator successfully dirties that node.
   *
   * <p>Warning: {@link #getReverseDepsUnsafe()} may return a live view of the reverse deps
   * collection of the marked-dirty node. The consumer of this data must be careful only to iterate
   * over and consume its values while that collection is guaranteed not to change. This is true
   * during invalidation, because reverse deps don't change during invalidation.
   */
  class MarkedDirtyResult {
    private final Iterable<SkyKey> reverseDepsUnsafe;

    public MarkedDirtyResult(Iterable<SkyKey> reverseDepsUnsafe) {
      this.reverseDepsUnsafe = Preconditions.checkNotNull(reverseDepsUnsafe);
    }

    public Iterable<SkyKey> getReverseDepsUnsafe() {
      return reverseDepsUnsafe;
    }
  }
}
