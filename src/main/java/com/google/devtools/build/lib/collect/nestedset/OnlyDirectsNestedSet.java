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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;

import java.util.Arrays;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Memory-optimized NestedSet implementation for NestedSets without transitive dependencies.
 *
 */
abstract class OnlyDirectsNestedSet<E> extends NestedSet<E> {

  private static final NestedSet[] EMPTY = new NestedSet[0];
  private final Object[] directDeps;

  public OnlyDirectsNestedSet(Object[] directDeps) { this.directDeps = directDeps; }

  @Override
  public abstract Order getOrder();

  @Override
  Object[] directMembers() {
    return directDeps;
  }

  @Override
  NestedSet[] transitiveSets() {
    return EMPTY;
  }

  @Override
  public boolean isEmpty() {
    return false;
  }

  /**
   * Currently all the Order implementations return the direct elements in the same order if they
   * do not have transitive elements. So we skip calling order.getExpander()... and return a view
   * of the array.
   */
  @SuppressWarnings("unchecked")
  @Override
  public List<E> toList() {
    return (List<E>) ImmutableList.copyOf(directDeps);
  }

  @SuppressWarnings("unchecked")
  @Override
  public Set<E> toSet() {
    return (Set<E>) ImmutableSet.copyOf(directDeps);
  }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    if (other == null) {
      return false;
    }
    return getOrder().equals(other.getOrder())
        && other instanceof OnlyDirectsNestedSet
        && Arrays.equals(directDeps, ((OnlyDirectsNestedSet) other).directDeps);
  }

  @Override
  public int shallowHashCode() {
    return Arrays.hashCode(directDeps);
  }
}
