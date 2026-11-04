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
package com.google.devtools.build.lib.collect.nestedset;

import static com.google.common.base.Preconditions.checkState;

import com.google.common.base.MoreObjects;
import javax.annotation.Nullable;

/**
 * An ambient scope that gives the {@link NestedSet}s created on the current thread a deterministic
 * identity: an owner (e.g. the key of the configured target, aspect, merged aspects, configuration
 * fragment, action template expansion, or {@code .bzl} file being evaluated) and an index that
 * counts the new non-empty sets created in the scope.
 *
 * <p>Several scopes may have the same owner only if they run the same deterministic computation,
 * e.g. merging the same aspects into the same target, so that their sets with the same index are
 * equal.
 *
 * <p>Scopes must not be nested: entering a scope while another one is active on the same thread
 * fails. The evaluation of a scoped unit never starts another one on the same thread, and if it did
 * (e.g. through work-stealing in the Skyframe thread pool), unrelated work would run inside this
 * scope and consume its indices.
 */
public final class NestedSetOwnerScope implements AutoCloseable {
  private static final ThreadLocal<NestedSetOwnerScope> current = new ThreadLocal<>();

  private final Object owner;
  private int nextIndex = 0;

  private NestedSetOwnerScope(Object owner) {
    this.owner = owner;
  }

  /**
   * Enters a new scope whose nested sets are owned by {@code owner}.
   *
   * @return an {@link AutoCloseable} scope that clears the scope upon closing
   * @throws IllegalStateException if a scope is already active on the current thread
   */
  public static NestedSetOwnerScope enter(Object owner) {
    NestedSetOwnerScope active = current.get();
    checkState(
        active == null,
        "NestedSetOwnerScope already active on this thread (%s), cannot enter for %s",
        active,
        owner);
    NestedSetOwnerScope scope = new NestedSetOwnerScope(owner);
    current.set(scope);
    return scope;
  }

  /** Returns the scope active on this thread, or {@code null} if none. */
  @Nullable
  static NestedSetOwnerScope current() {
    return current.get();
  }

  /**
   * Returns the number of newly constructed non-empty {@link NestedSet} instances in the scope
   * active on this thread, or 0 if no scope is active.
   */
  public static int getCurrentNestedSetCount() {
    NestedSetOwnerScope scope = current.get();
    return scope != null ? scope.nextIndex : 0;
  }

  Object getOwner() {
    return owner;
  }

  /** Returns the index for a new non-empty nested set. */
  int nextIndex() {
    return nextIndex++;
  }

  @Override
  public void close() {
    current.remove();
  }

  @Override
  public String toString() {
    return MoreObjects.toStringHelper(this)
        .add("owner", owner)
        .add("nextIndex", nextIndex)
        .toString();
  }
}
