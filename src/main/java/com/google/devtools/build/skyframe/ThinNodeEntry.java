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
   * Returns true if the entry is marked dirty, meaning that at least one of its transitive
   * dependencies is marked changed.
   */
  @ThreadSafe
  boolean isDirty();

  /**
   * Returns true if the entry is marked changed, meaning that it must be re-evaluated even if its
   * dependencies' values have not changed.
   */
  @ThreadSafe
  boolean isChanged();

  /**
   * Marks this node dirty, or changed if {@code isChanged} is true.
   *
   * <p>A dirty node P is re-evaluated during the evaluation phase if it's requested and directly
   * depends on some node C whose value changed since the last evaluation of P. If it's requested
   * and there is no such node C, P is marked clean.
   *
   * <p>A changed node is re-evaluated during the evaluation phase if it's requested (regardless of
   * the state of its dependencies).
   *
   * @return a {@link MarkedDirtyResult} indicating whether the call was redundant and which may
   *     include the node's reverse deps
   */
  @ThreadSafe
  MarkedDirtyResult markDirty(boolean isChanged) throws InterruptedException;

  /** Returned by {@link #markDirty}. */
  interface MarkedDirtyResult {

    /** Returns true iff the node was clean prior to the {@link #markDirty} call. */
    boolean wasClean();

    /**
     * Returns true iff the call to {@link #markDirty} was the same as some previous call to {@link
     * #markDirty} (i.e., sharing the same {@code isChanged} parameter value) since the last time
     * the node was clean.
     *
     * <p>More specifically, this returns true iff the call was {@code n.markDirty(b)} and prior to
     * the call {@code n.isDirty() && n.isChanged() == b}).
     */
    boolean wasCallRedundant();

    /**
     * If {@code wasClean()}, this returns an iterable of the node's reverse deps for efficiency,
     * because the {@link #markDirty} caller may be doing graph invalidation, and after dirtying a
     * node, the invalidation process may want to dirty the node's reverse deps.
     *
     * <p>If {@code !wasClean()}, this must not be called. It will throw {@link
     * IllegalStateException}.
     *
     * <p>Warning: the returned iterable may be a live view of the reverse deps collection of the
     * marked-dirty node. The consumer of this data must be careful only to iterate over and consume
     * its values while that collection is guaranteed not to change. This is true during
     * invalidation, because reverse deps don't change during invalidation.
     */
    Iterable<SkyKey> getReverseDepsUnsafeIfWasClean();
  }

  /** A {@link MarkedDirtyResult} returned when {@link #markDirty} is called on a clean node. */
  class FromCleanMarkedDirtyResult implements MarkedDirtyResult {
    private final Iterable<SkyKey> reverseDepsUnsafe;

    public FromCleanMarkedDirtyResult(Iterable<SkyKey> reverseDepsUnsafe) {
      this.reverseDepsUnsafe = Preconditions.checkNotNull(reverseDepsUnsafe);
    }

    @Override
    public boolean wasClean() {
      return true;
    }

    @Override
    public boolean wasCallRedundant() {
      return false;
    }

    @Override
    public Iterable<SkyKey> getReverseDepsUnsafeIfWasClean() {
      return reverseDepsUnsafe;
    }
  }

  /** A {@link MarkedDirtyResult} returned when {@link #markDirty} is called on a dirty node. */
  class FromDirtyMarkedDirtyResult implements MarkedDirtyResult {
    static final FromDirtyMarkedDirtyResult REDUNDANT = new FromDirtyMarkedDirtyResult(true);
    static final FromDirtyMarkedDirtyResult NOT_REDUNDANT = new FromDirtyMarkedDirtyResult(false);

    private final boolean redundant;

    private FromDirtyMarkedDirtyResult(boolean redundant) {
      this.redundant = redundant;
    }

    @Override
    public boolean wasClean() {
      return false;
    }

    @Override
    public boolean wasCallRedundant() {
      return redundant;
    }

    @Override
    public Iterable<SkyKey> getReverseDepsUnsafeIfWasClean() {
      throw new IllegalStateException();
    }
  }
}
