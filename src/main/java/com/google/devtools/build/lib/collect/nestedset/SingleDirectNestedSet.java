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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.common.collect.Iterators;

import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * Memory-efficient implementation for nested sets with one element.
 */
public abstract class SingleDirectNestedSet<E> extends NestedSet<E> {

  private static final NestedSet[] EMPTY = new NestedSet[0];
  private final E e;

  public SingleDirectNestedSet(E e) { this.e = Preconditions.checkNotNull(e); }

  @Override
  public Iterator<E> iterator() { return Iterators.singletonIterator(e); }

  @Override
  Object[] directMembers() { return new Object[]{e}; }

  @Override
  NestedSet[] transitiveSets() { return EMPTY; }

  @Override
  public boolean isEmpty() { return false; }

    @Override
  public List<E> toList() { return ImmutableList.of(e); }

  @Override
  public Set<E> toSet() { return ImmutableSet.of(e); }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    return other instanceof SingleDirectNestedSet
        && e.equals(((SingleDirectNestedSet) other).e);
  }

  @Override
  public int shallowHashCode() {
    return e.hashCode();
  }
}
