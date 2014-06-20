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

import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;

import java.util.Collection;
import java.util.LinkedHashSet;

/**
 * A builder for nested sets.
 *
 * <p>The builder supports the standard builder interface (that is, {@code #add}, {@code #addAll}
 * and {@code #addTransitive} followed by {@code build}), in addition to shortcut methods
 * {@code #wrap} and {@code #of}.
 */
public final class NestedSetBuilder<E> {

  private final Order order;
  private final Collection<E> items = new LinkedHashSet<>();
  private final Collection<NestedSet<E>> transitiveSets = new LinkedHashSet<>();

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
  public NestedSetBuilder<E> addAll(Iterable<E> elements) {
    Preconditions.checkNotNull(elements);
    Iterables.addAll(items, elements);
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
   * of this set. Either they must match or this set must be a {@code STABLE_ORDER} set.
   *
   * @return the builder.
   */
  public NestedSetBuilder<E> addTransitive(NestedSet<E> subset) {
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
  public NestedSet<E> build() {
    if (isEmpty()) {
      return order.<E>emptySet();
    }
    if (items.isEmpty() && (transitiveSets.size() == 1)) {
      NestedSet<E> candidate = Iterables.getOnlyElement(transitiveSets);
      if (candidate.getOrder().equals(order)) {
        return candidate;
      }
    }
    return order.createNestedSet(ImmutableList.copyOf(items), ImmutableList.copyOf(transitiveSets));
  }

  /**
   * Creates a nested set from a given list of items.
   *
   * <p>If the list of items is an {@link ImmutableList}, reuses the list as the backing store for
   * the nested set.
   */
  public static <E> NestedSet<E> wrap(Order order, Iterable<E> wrappedItems) {
    ImmutableList<E> wrappedList = ImmutableList.copyOf(wrappedItems);
    if (wrappedList.isEmpty()) {
      return order.<E>emptySet();
    }
    return order.createNestedSet(wrappedList, ImmutableList.<NestedSet<E>>of());
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
}
