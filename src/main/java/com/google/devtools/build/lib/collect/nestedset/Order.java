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

import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Maps;

import java.util.HashMap;

/**
 * Type of a nested set (defines order).
 *
 *
 * <p>STABLE_ORDER: an unspecified traversal order.  Use when the order of elements does not matter.
 *
 *
 * <p>COMPILE_ORDER: left-to-right postorder.
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "A C B D"
 * (child-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of nested sets go before
 * the direct members of a set, for example in the case of Javascript dependencies.
 *
 *
 * <p>LINK_ORDER: a variation of left-to-right preorder.
 *
 * <p>For example, for the nested set {A, C, {B, D}}, the iteration order is "A C B D"
 * (parent-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of
 * nested sets go after the direct members of a set, for example when providing
 * a list of libraries to the C++ compiler.
 *
 * <p>The custom ordering has the property that elements of nested sets always come
 * before elements of descendant nested sets. Left-to-right order is preserved if
 * possible, both for items and for references to nested sets.
 *
 * <p>The left-to-right pre-order-like ordering is implemented by running a
 * right-to-left postorder traversal and then reversing the result.
 *
 * <p>The reason naive left-to left-to-right preordering is not used here is that
 * it does not handle diamond-like structures properly. For example, take the
 * following structure (nesting downwards):
 *
 * <pre>
 *    A
 *   / \
 *  B   C
 *   \ /
 *    D
 * </pre>
 *
 * <p>Naive preordering would produce "A B D C", which does not preserve the
 * "parent before child" property: C is a parent of D, so C should come before
 * D. Either "A B C D" or "A C B D" would be acceptable. This implementation
 * returns the first option of the two so that left-to-right order is preserved.
 *
 * <p>In case the nested sets form a tree, the ordering algorithm is equivalent to
 * standard left-to-right preorder.
 *
 * <p>Sometimes it may not be possible to preserve left-to-right order:
 *
 * <pre>
 *      A
 *    /   \
 *   B     C
 *  / \   / \
 *  \   E   /
 *   \     /
 *    \   /
 *      D
 * </pre>
 *
 * <p>The left branch (B) would indicate "D E" ordering and the right branch (C)
 * dictates "E D". In such cases ordering is decided by the rightmost branch
 * because of the list reversing behind the scenes, so the ordering in the final
 * enumeration will be "E D".
 *
 *
 * <p>NAIVE_LINK_ORDER: a left-to-right preordering.
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "B D A C".
 *
 * <p>The order is called naive because it does no special treatment of dependency graphs
 * that are not trees. For such graphs the property of parent-before-dependencies in the iteration
 * order will not be upheld. For example, the diamond-shape graph A->{B, C}, B->{D}, C->{D} will be
 * enumerated as "A B D C" rather than "A B C D" or "A C B D".
 *
 * <p>The difference from LINK_ORDER is that this order gives priority to
 * left-to-right order over dependencies-after-parent ordering. Note that the latter is usually more
 * important, so please use LINK_ORDER whenever possible.
 */
public enum Order {
  STABLE_ORDER("stable"),
  COMPILE_ORDER("compile"),
  LINK_ORDER("link"),
  NAIVE_LINK_ORDER("naive_link");

  private static final ImmutableMap<String, Order> VALUES;

  private final String name;
  private final NestedSet<?> emptySet;

  private Order(String name) {
    this.name = name;
    this.emptySet = new NestedSet<>(this);
  }

  /**
   * Returns an empty set of the given ordering.
   */
  @SuppressWarnings("unchecked")  // Nested sets are immutable, so a downcast is fine.
  <E> NestedSet<E> emptySet() {
    return (NestedSet<E>) emptySet;
  }

  public String getName() {
    return name;
  }

  /**
   * Parses the given string as a set order
   *
   * @param name Unique name of the order
   * @return Order The appropriate order instance
   * @throws IllegalArgumentException If the name is not valid
   */
  public static Order parse(String name) {
    if (!VALUES.containsKey(name)) {
      throw new IllegalArgumentException("Invalid order: " + name);
    }

    return VALUES.get(name);
  }

  /**
   * Indexes all possible values by name and stores the results in a {@code ImmutableMap}
   */
  static {
    Order[] tmpValues = Order.values();

    HashMap<String, Order> entries = Maps.newHashMapWithExpectedSize(tmpValues.length);

    for (Order current : tmpValues) {
      entries.put(current.getName(), current);
    }

    VALUES = ImmutableMap.copyOf(entries);
  }
}
