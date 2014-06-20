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
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableCollection;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableSet;
import com.google.devtools.build.lib.concurrent.ThreadSafety.ThreadCompatible;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.Objects;
import java.util.Set;

import javax.annotation.Nullable;

/**
 * A base class for an in-memory implementation of the {@code NestedSet} interface.
 *
 * <p>This class could easily be concrete. We do a micro-optimization here by storing the order
 * enum on the class itself rather than on every instance, thereby saving 8 bytes per instance.
 */
abstract class InMemoryNestedSet<E> implements NestedSet<E> {

  private final ImmutableList<E> items;
  private final ImmutableList<NestedSet<E>> transitiveSets;
  private Object memo;

  InMemoryNestedSet(ImmutableList<E> items, ImmutableList<NestedSet<E>> transitiveSets) {
    this.items = Preconditions.checkNotNull(items);
    this.transitiveSets = Preconditions.checkNotNull(transitiveSets);
    Preconditions.checkState(!(items.isEmpty() && transitiveSets.isEmpty()));
  }

  @Override
  public abstract Order getOrder();

  private final class NestedIterator implements Iterator<E> {
    private Iterator<E> delegate = null;
    @Override
    public boolean hasNext() {
      if (delegate == null) {
        return !isEmpty();
      }
      return delegate.hasNext();
    }

    @Override
    public E next() {
      if (delegate == null) {
        delegate = toCollection().iterator();
      }
      return delegate.next();
    }

    @Override
    public void remove() {
      throw new UnsupportedOperationException();
    }
  }
  // Concurrent access should be synchronized, since a delegate field is initialized lazily.
  @ThreadCompatible
  @Override
  public Iterator<E> iterator() {
    return new NestedIterator();
  }

  @Override
  public ImmutableList<E> directMembers() {
    return items;
  }

  @Override
  public ImmutableList<NestedSet<E>> transitiveSets() {
    return transitiveSets;
  }

  @Override
  public boolean isEmpty() {
    return false;
  }

  @Override
  public List<E> toList() {
    if (transitiveSets.isEmpty()) {
      return items;
    }

    ImmutableList.Builder<E> builder = ImmutableList.builder();
    fill(builder);
    return builder.build();
  }

  @Override
  public Collection<E> toCollection() {
    return toList();
  }

  @Override
  public Set<E> toSet() {
    ImmutableSet.Builder<E> builder = ImmutableSet.builder();
    fill(builder);
    return builder.build();
  }

  private void fill(ImmutableCollection.Builder<E> builder) {
    Uniqueifier memoed;
    synchronized (this) {
      if (memo == null) {
        RecordingUniqueifier uniqueifier = new RecordingUniqueifier();
        fill(builder, uniqueifier);
        memo = uniqueifier.getMemo();
        return;
      } else {
        memoed = RecordingUniqueifier.createReplayUniqueifier(memo);
      }
    }
    fill(builder, memoed);
  }

  private void fill(ImmutableCollection.Builder<E> builder, Uniqueifier uniqueifier) {
    getOrder().<E>expander().expandInto(this, uniqueifier, builder);
  }

  @Override
  public boolean shallowEquals(@Nullable NestedSet<? extends E> other) {
    if (this == other) {
      return true;
    }
    if (other == null) {
      return false;
    }
    if (!getOrder().equals(other.getOrder())) {
      return false;
    }
    if (!directMembers().equals(other.directMembers())) {
      return false;
    }
    // All this code just checks whether transitiveSets() is the same as other.transitiveSets(),
    // comparing elements with ==, in the same order.
    Iterator<NestedSet<E>> myTransIter = transitiveSets().iterator();
    Iterator<? extends NestedSet<? extends E>> otherTransIter =
        other.transitiveSets().iterator();
    while (true) {
      if (myTransIter.hasNext() != otherTransIter.hasNext()) {
        return false;
      }
      if (!myTransIter.hasNext()) {
        break;  // The transitiveSets are equal.
      }
      NestedSet<E> myNested = myTransIter.next();
      NestedSet<? extends E> otherNested = otherTransIter.next();
      // Use reference equality instead of recursively checking both inner sets.
      // This is "pessimistic" in that it may fail to notice that the inner sets
      // have identical elements, but takes less time.
      if (myNested != otherNested) {
        return false;
      }
    }
    return true;
  }

  @Override
  public int shallowHashCode() {
    return Objects.hash(getOrder(), directMembers(), transitiveSets());
  }

  @Override
  public String toString() {
    String members = Joiner.on(", ").join(directMembers());
    String nestedSets = Joiner.on(", ").join(transitiveSets());
    String separator = members.length() > 0 && nestedSets.length() > 0 ? ", " : "";
    return "{" + members + separator + nestedSets + "}";
  }
}
