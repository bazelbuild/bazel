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

package com.google.devtools.build.lib.collect;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.devtools.build.lib.collect.nestedset.NestedSet;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * An immutable chain of immutable Iterables.
 *
 * <p>This class is defined for the sole purpose of being able to check for immutability (using
 * instanceof). Otherwise, we could use a plain Iterable (as returned by {@code
 * Iterables.concat()}).
 *
 * @see CollectionUtils#checkImmutable(Iterable)
 */
public final class IterablesChain<T> implements Iterable<T> {
  public static <T> Iterable<T> concat(Iterable<? extends T> a, Iterable<? extends T> b) {
    return IterablesChain.<T>builder().add(a).add(b).build();
  }

  private final Iterable<T> chain;

  IterablesChain(Iterable<T> chain) {
    this.chain = chain;
  }

  @Override
  public Iterator<T> iterator() {
    return chain.iterator();
  }

  public static <T> Builder<T> builder() {
    return new Builder<>();
  }

  @Override
  public String toString() {
    return "[" + Joiner.on(", ").join(this) + "]";
  }

  /** Builder for IterablesChain. */
  public static class Builder<T> {
    private List<Iterable<? extends T>> iterables = new ArrayList<>();

    private Builder() {}

    /**
     * Adds an immutable iterable to the end of the chain.
     *
     * <p>If the iterable can not be confirmed to be immutable, a runtime error is thrown.
     */
    public Builder<T> add(Iterable<? extends T> iterable) {
      CollectionUtils.checkImmutable(iterable);
      // Avoid unnecessarily expanding a NestedSet.
      boolean isEmpty =
          iterable instanceof NestedSet
              ? ((NestedSet<?>) iterable).isEmpty()
              : Iterables.isEmpty(iterable);
      if (!isEmpty) {
        iterables.add(iterable);
      }
      return this;
    }

    /** Adds a single element to the chain. */
    public Builder<T> addElement(T element) {
      iterables.add(ImmutableList.of(element));
      return this;
    }

    /** Returns true if the chain is empty. */
    public boolean isEmpty() {
      return iterables.isEmpty();
    }

    /** Builds an iterable that iterates through all elements in this chain. */
    @SuppressWarnings("unchecked")
    public Iterable<T> build() {
      int size = iterables.size();
      if (size == 0) {
        return ImmutableList.of();
      } else if (size == 1) {
        // cast is type safe since Iterable<T> doesn't allow mutation
        return (Iterable<T>) iterables.get(0);
      }
      return new IterablesChain<>(Iterables.concat(ImmutableList.copyOf(iterables)));
    }
  }
}
