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

import static com.google.common.collect.Iterables.getOnlyElement;
import static com.google.common.collect.Iterables.isEmpty;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.MapMaker;
import com.google.devtools.build.lib.collect.CompactHashSet;
import com.google.devtools.build.lib.util.Preconditions;

import java.util.concurrent.ConcurrentMap;

/**
 * A builder for nested sets.
 *
 * <p>The builder supports the standard builder interface (that is, {@code #add}, {@code #addAll}
 * and {@code #addTransitive} followed by {@code build}), in addition to shortcut method
 * {@code #wrap}.
 */
public final class NestedSetBuilder<E> {

  private final Order order;
  private final CompactHashSet<E> items = CompactHashSet.create();
  private final CompactHashSet<NestedSet<? extends E>> transitiveSets = CompactHashSet.create();

  public NestedSetBuilder(Order order) {
    this.order = order;
  }

  /** Returns whether the set to be built is empty. */
  public boolean isEmpty() {
    return items.isEmpty() && transitiveSets.isEmpty();
  }

  /**
   * Add an element.
   *
   * <p>Preserves ordering of added elements. Discards duplicate values.
   * Throws an exception if a null value is passed in.
   *
   * <p>The collections of the direct members of the set and the nested sets are
   * kept separate, so the order between multiple add/addAll calls matters,
   * and the order between multiple addTransitive calls matters, but the order
   * between add/addAll and addTransitive does not.
   *
   * @return the builder.
   */
  @SuppressWarnings("unchecked")  // B is the type of the concrete subclass
  public NestedSetBuilder<E> add(E element) {
    Preconditions.checkNotNull(element);
    items.add(element);
    return this;
  }

  /**
   * Adds a collection of elements to the set.
   *
   * <p>This is equivalent to invoking {@code add} for every item of the collection in iteration
   * order.
   *
   *  <p>The collections of the direct members of the set and the nested sets are kept separate, so
   * the order between multiple add/addAll calls matters, and the order between multiple
   * addTransitive calls matters, but the order between add/addAll and addTransitive does not.
   *
   * @return the builder.
   */
  @SuppressWarnings("unchecked")  // B is the type of the concrete subclass
  public NestedSetBuilder<E> addAll(Iterable<? extends E> elements) {
    Preconditions.checkNotNull(elements);
    Iterables.addAll(items, elements);
    return this;
  }

  /**
   * @deprecated Use {@link #addTransitive} to avoid excessive memory use.
   */
  @Deprecated
  public NestedSetBuilder<E> addAll(NestedSet<E> elements) {
    // Do not delete this method, or else addAll(Iterable) calls with a NestedSet argument
    // will not be flagged.
    Iterable<E> it = elements;
    addAll(it);
    return this;
  }

  /**
   * Adds another nested set to this set.
   *
   *  <p>Preserves ordering of added nested sets. Discards duplicate values. Throws an exception if
   * a null value is passed in.
   *
   *  <p>The collections of the direct members of the set and the nested sets are kept separate, so
   * the order between multiple add/addAll calls matters, and the order between multiple
   * addTransitive calls matters, but the order between add/addAll and addTransitive does not.
   *
   * <p>An error will be thrown if the ordering of {@code subset} is incompatible with the ordering
   * of this set. Either they must match or one must be a {@code STABLE_ORDER} set.
   *
   * @return the builder.
   */
  public NestedSetBuilder<E> addTransitive(NestedSet<? extends E> subset) {
    Preconditions.checkNotNull(subset);
    if (subset.getOrder() != order && order != Order.STABLE_ORDER
            && subset.getOrder() != Order.STABLE_ORDER) {
      // Note that this check is not strictly necessary, although keeping the nested set types
      // consistent helps readability and protects against bugs. The polymorphism regarding
      // STABLE_ORDER is allowed in order to be able to, e.g., include an arbitrary nested set in
      // the inputs of an action, or include a nested set that is indifferent to its order in
      // multiple nested sets.
      throw new IllegalStateException(subset.getOrder() + " != " + order);
    }
    if (!subset.isEmpty()) {
      transitiveSets.add(subset);
    }
    return this;
  }

  /**
   * Builds the actual nested set.
   *
   * <p>This method may be called multiple times with interleaved {@link #add}, {@link #addAll} and
   * {@link #addTransitive} calls.
   */
  // Casting from CompactHashSet<NestedSet<? extends E>> to CompactHashSet<NestedSet<E>> by way of
  // CompactHashSet<?>.
  @SuppressWarnings("unchecked")
  public NestedSet<E> build() {
    if (isEmpty()) {
      return order.emptySet();
    }

    // This cast is safe because NestedSets are immutable -- we will never try to add an element to
    // these nested sets, only to retrieve elements from them. Thus, treating them as NestedSet<E>
    // is safe.
    CompactHashSet<NestedSet<E>> transitiveSetsCast =
        (CompactHashSet<NestedSet<E>>) (CompactHashSet<?>) transitiveSets;
    if (items.isEmpty() && (transitiveSetsCast.size() == 1)) {
      NestedSet<E> candidate = getOnlyElement(transitiveSetsCast);
      if (candidate.getOrder().equals(order)) {
        return candidate;
      }
    }
    return new NestedSet<E>(order, items, transitiveSetsCast);
  }

  private static final ConcurrentMap<ImmutableList<?>, NestedSet<?>> immutableListCache =
      new MapMaker().weakKeys().makeMap();

  /**
   * Creates a nested set from a given list of items.
   */
  @SuppressWarnings("unchecked")
  public static <E> NestedSet<E> wrap(Order order, Iterable<E> wrappedItems) {
    ImmutableList<E> wrappedList = ImmutableList.copyOf(wrappedItems);
    if (wrappedList.isEmpty()) {
      return order.emptySet();
    } else if (order == Order.STABLE_ORDER
               && wrappedList == wrappedItems && wrappedList.size() > 1) {
      NestedSet<?> cached = immutableListCache.get(wrappedList);
      if (cached != null) {
        return (NestedSet<E>) cached;
      }
      NestedSet<E> built = new NestedSetBuilder<E>(order).addAll(wrappedList).build();
      immutableListCache.putIfAbsent(wrappedList, built);
      return built;
    } else {
      return new NestedSetBuilder<E>(order).addAll(wrappedList).build();
    }
  }

  /**
   * Creates a nested set with the given list of items as its elements.
   */
  @SuppressWarnings("unchecked")
  public static <E> NestedSet<E> create(Order order, E... elems) {
    return wrap(order, ImmutableList.copyOf(elems));
  }

  /**
   * Creates an empty nested set.
   */
  public static <E> NestedSet<E> emptySet(Order order) {
    return order.emptySet();
  }

  /**
   * Creates a builder for stable order nested sets.
   */
  public static <E> NestedSetBuilder<E> stableOrder() {
    return new NestedSetBuilder<>(Order.STABLE_ORDER);
  }

  /**
   * Creates a builder for compile order nested sets.
   */
  public static <E> NestedSetBuilder<E> compileOrder() {
    return new NestedSetBuilder<>(Order.COMPILE_ORDER);
  }

  /**
   * Creates a builder for link order nested sets.
   */
  public static <E> NestedSetBuilder<E> linkOrder() {
    return new NestedSetBuilder<>(Order.LINK_ORDER);
  }

  /**
   * Creates a builder for naive link order nested sets.
   */
  public static <E> NestedSetBuilder<E> naiveLinkOrder() {
    return new NestedSetBuilder<>(Order.NAIVE_LINK_ORDER);
  }

  public static <E> NestedSetBuilder<E> fromNestedSet(NestedSet<E> set) {
    return new NestedSetBuilder<E>(set.getOrder()).addTransitive(set);
  }

  /**
   * Creates a Builder with the contents of 'sets'.
   *
   * <p>If 'sets' is empty, a stable-order empty NestedSet is returned.
   */
  public static <E> NestedSetBuilder<E> fromNestedSets(Iterable<NestedSet<E>> sets) {
    NestedSet<?> firstSet = Iterables.getFirst(sets, null /* defaultValue */);
    if (firstSet == null) {
      return stableOrder();
    }
    NestedSetBuilder<E> result = new NestedSetBuilder<>(firstSet.getOrder());
    for (NestedSet<E> set : sets) {
      result.addTransitive(set);
    }
    return result;
  }
}
