// Copyright 2015 Google Inc. All rights reserved.
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

import javax.annotation.Nullable;

/**
 * A node in the graph without the means to access its value. All operations on this class are
 * thread-safe.
 *
 * <p>This interface is public only for the benefit of alternative graph implementations outside of
 * the package.
 */
public interface ThinNodeEntry {

  /** Returns whether the entry has been built and is finished evaluating. */
  @ThreadSafe
  boolean isDone();

  /**
   * Returns an immutable iterable of the direct deps of this node. This method may only be called
   * after the evaluation of this node is complete.
   *
   * <p>This method is not very efficient, but is only be called in limited circumstances --
   * when the node is about to be deleted, or when the node is expected to have no direct deps (in
   * which case the overhead is not so bad). It should not be called repeatedly for the same node,
   * since each call takes time proportional to the number of direct deps of the node.
   */
  @ThreadSafe
  Iterable<SkyKey> getDirectDeps();

  /**
   * Removes a reverse dependency.
   */
  @ThreadSafe
  void removeReverseDep(SkyKey reverseDep);

  /**
   * Removes a reverse dependency.
   *
   * <p>May only be called if this entry is not done (i.e. {@link #isDone} is false) and
   * {@param reverseDep} is present in {@link #getReverseDeps}
   */
  @ThreadSafe
  void removeInProgressReverseDep(SkyKey reverseDep);

  /**
   * Returns a copy of the set of reverse dependencies. Note that this introduces a potential
   * check-then-act race; {@link #removeReverseDep} may fail for a key that is returned here.
   */
  @ThreadSafe
  Iterable<SkyKey> getReverseDeps();

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
   * Marks this node dirty, or changed if {@code isChanged} is true. The node is put in the
   * just-created state. It will be re-evaluated if necessary during the evaluation phase,
   * but if it has not changed, it will not force a re-evaluation of its parents.
   *
   * <p>{@code markDirty(b)} must not be called on an undone node if {@code isChanged() == b}.
   * It is the caller's responsibility to ensure that this does not happen.  Calling
   * {@code markDirty(false)} when {@code isChanged() == true} has no effect. The idea here is that
   * the caller will only ever want to call {@code markDirty()} a second time if a transition from a
   * dirty-unchanged state to a dirty-changed state is required.
   *
   * @return true if the node was previously clean, and false if it was already dirty. If it was
   * already dirty, the caller should abort its handling of this node, since another thread is
   * already dealing with it.
   */
  @Nullable
  @ThreadSafe
  boolean markDirty(boolean isChanged);
}
