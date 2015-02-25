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
 * Memory-optimized NestedSet implementation for NestedSets with one direct element and
 * many transitive dependencies.
 */
abstract class OneDirectManyTransitive<E> extends MemoizedUniquefierNestedSet<E> {

  private final Object direct;
  private final NestedSet[] transitives;
  private Object memo;

  OneDirectManyTransitive(Object direct, NestedSet[] transitives) {
    this.direct = direct;
    this.transitives = transitives;
  }

  @Override
  Object getMemo() { return memo; }

  @Override
  void setMemo(Object memo) { this.memo = memo; }

  @Override
  Object[] directMembers() { return new Object[]{direct}; }

  @Override
  NestedSet[] transitiveSets() { return transitives; }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    return other != null
        && getOrder().equals(other.getOrder())
        && other instanceof OneDirectManyTransitive
        && direct.equals(((OneDirectManyTransitive) other).direct)
        && Arrays.equals(transitives, ((OneDirectManyTransitive) other).transitives);
  }

  @Override
  public int shallowHashCode() {
    return Objects.hash(getOrder(), direct, Arrays.hashCode(transitives)); }
}
