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

import com.google.common.base.Joiner;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A list-like iterable that supports efficient nesting.
 *
 * @see NestedSetBuilder
 */
public abstract class NestedSet<E> implements Iterable<E>, Serializable {

  NestedSet() {}

  /**
   * Returns the ordering of this nested set.
   */
  public abstract Order getOrder();

  /**
   * Returns a collection of elements added to this specific set in an implementation-specified
   * order.
   *
   * <p>Elements from subsets are not taken into account.
   *
   * <p>The reason for using Object[] instead of E[] is that when we build the NestedSet we
   * would need to have access to the specific class that E represents in order to create an E
   * array. Since this method is only designed to be used internally it is fine to keep it as
   * Object[].
   *
   * <p>Callers of this method should only consume the objects and not modify the array.
   */
  abstract Object[] directMembers();

  /**
   * Returns the collection of sets included as subsets in this set.
   *
   * <p>Callers of this method should only consume the objects and not modify the array.
   */
  abstract NestedSet[] transitiveSets();

  /**
   * Returns true if the set is empty.
   */
  public abstract boolean isEmpty();

  /**
   * Returns a collection of all unique elements of this set (including subsets)
   * in an implementation-specified order as a {@code Collection}.
   *
   * <p>If you do not need a Collection and an Iterable is enough, use the
   * nested set itself as an Iterable.
   */
  public Collection<E> toCollection() {
    return toList();
  }

  /**
   * Returns a collection of all unique elements of this set (including subsets)
   * in an implementation-specified order as a {code List}.
   *
   * <p>Use {@link #toCollection} when possible for better efficiency.
   */
  public abstract List<E> toList();

  /**
   * Returns a collection of all unique elements of this set (including subsets)
   * in an implementation-specified order as a {@code Set}.
   *
   * <p>Use {@link #toCollection} when possible for better efficiency.
   */
  public abstract Set<E> toSet();

  /**
   * Returns true if this set is equal to {@code other} based on the top-level
   * elements and object identity (==) of direct subsets.  As such, this function
   * can fail to equate {@code this} with another {@code NestedSet} that holds
   * the same elements.  It will never fail to detect that two {@code NestedSet}s
   * are different, however.
   *
   * @param other the {@code NestedSet} to compare against.
   */
  public abstract boolean shallowEquals(@Nullable NestedSet<? extends E> other);

  /**
   * Returns a hash code that produces a notion of identity that is consistent with
   * {@link #shallowEquals}. In other words, if two {@code NestedSet}s are equal according
   * to {@code #shallowEquals}, then they return the same {@code shallowHashCode}.
   *
   * <p>The main reason for having these separate functions instead of reusing
   * the standard equals/hashCode is to minimize accidental use, since they are
   * different from both standard Java objects and collection-like objects.
   */
  public abstract int shallowHashCode();

  @Override
  public String toString() {
    String members = Joiner.on(", ").join(directMembers());
    String nestedSets = Joiner.on(", ").join(transitiveSets());
    String separator = members.length() > 0 && nestedSets.length() > 0 ? ", " : "";
    return "{" + members + separator + nestedSets + "}";
  }

  @Override
  public Iterator<E> iterator() { return new NestedSetLazyIterator<>(this); }

}
