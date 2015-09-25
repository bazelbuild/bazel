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
package com.google.devtools.build.lib.collect.nestedset;

import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Memory-efficient implementation for the case where we only have one transitive NestedSet.
 *
 * <p>Note that we cannot simply delegate to the inner NestedSet because the order could be
 * different and the top-level order is the correct one.
 */
abstract class OnlyOneTransitiveNestedSet<E> extends MemoizedUniquefierNestedSet<E> {

  private static final Object[] EMPTY = new Object[0];

  private final NestedSet<E> transitive;
  private Object memo;

  public OnlyOneTransitiveNestedSet(NestedSet<E> transitive) {
    this.transitive = transitive;
  }

  @Override
  Object getMemo() { return memo; }

  @Override
  void setMemo(Object memo) { this.memo = memo; }

  @Override
  Object[] directMembers() {
    return EMPTY;
  }

  @Override
  NestedSet[] transitiveSets() {
    return new NestedSet[]{transitive};
  }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    return other != null
        && getOrder().equals(other.getOrder())
        && other instanceof OnlyOneTransitiveNestedSet
        && transitive == ((OnlyOneTransitiveNestedSet) other).transitive;
  }

  @Override
  public int shallowHashCode() {
    return Objects.hash(getOrder(), transitive);
  }
}
