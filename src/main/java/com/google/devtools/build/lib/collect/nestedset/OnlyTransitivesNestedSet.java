// Copyright 2014 Google Inc. All rights reserved.
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

import java.util.Arrays;
import java.util.Objects;

import javax.annotation.Nullable;

/**
 * Memory-efficient implementation for the case where we have one direct element and one
 * transitive NestedSet.
 */
abstract class OnlyTransitivesNestedSet<E> extends MemoizedUniquefierNestedSet<E> {

  private static final NestedSet[] EMPTY = new NestedSet[0];

  private final NestedSet[] transitives;
  private Object memo;

  OnlyTransitivesNestedSet(NestedSet[] transitives) {
    this.transitives = transitives;
  }

  @Override
  Object getMemo() { return memo; }

  @Override
  void setMemo(Object memo) { this.memo = memo; }

  @Override
  Object[] directMembers() { return EMPTY; }

  @Override
  NestedSet[] transitiveSets() { return transitives; }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    return other != null
        && getOrder().equals(other.getOrder())
        && other instanceof OnlyTransitivesNestedSet
        && Arrays.equals(transitives, ((OnlyTransitivesNestedSet) other).transitives);
  }

  @Override
  public int shallowHashCode() {
    return Objects.hash(getOrder(), Arrays.hashCode(transitives)); }
}
