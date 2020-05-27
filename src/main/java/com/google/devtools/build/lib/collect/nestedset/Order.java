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
import com.google.devtools.build.lib.skyframe.serialization.autocodec.AutoCodec;
import java.util.HashMap;

/**
 * Type of a nested set (defines order).
 *
 * <p>STABLE_ORDER: an unspecified traversal order. Use when the order of elements does not matter.
 * In Starlark it is called "default"; its older deprecated name is "stable".
 *
 * <p>COMPILE_ORDER: left-to-right postorder. In Starlark it is called "postorder"; its older
 * deprecated name is "compile".
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "A C B D"
 * (child-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of nested sets go before
 * the direct members of a set, for example in the case of Javascript dependencies.
 *
 * <p>LINK_ORDER: a variation of left-to-right preorder that enforces topological sorting. In
 * Starlark it is called "topological"; its older deprecated name is "link".
 *
 * <p>For example, for the nested set {A, C, {B, D}}, the iteration order is "A C B D"
 * (parent-first).
 *
 * <p>This type of set would typically be used for artifacts where elements of nested sets go after
 * the direct members of a set, for example when providing a list of libraries to the C++ compiler.
 *
 * <p>The custom ordering has the property that elements of nested sets always come before elements
 * of descendant nested sets. Left-to-right order is preserved if possible, both for items and for
 * references to nested sets.
 *
 * <p>The left-to-right pre-order-like ordering is implemented by running a right-to-left postorder
 * traversal and then reversing the result.
 *
 * <p>The reason naive left-to left-to-right preordering is not used here is that it does not handle
 * diamond-like structures properly. For example, take the following structure (nesting downwards):
 *
 * <pre>
 *    A
 *   / \
 *  B   C
 *   \ /
 *    D
 * </pre>
 *
 * <p>Naive preordering would produce "A B D C", which does not preserve the "parent before child"
 * property: C is a parent of D, so C should come before D. Either "A B C D" or "A C B D" would be
 * acceptable. This implementation returns the first option of the two so that left-to-right order
 * is preserved.
 *
 * <p>In case the nested sets form a tree, the ordering algorithm is equivalent to standard
 * left-to-right preorder.
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
 * <p>The left branch (B) would indicate "D E" ordering and the right branch (C) dictates "E D". In
 * such cases ordering is decided by the rightmost branch because of the list reversing behind the
 * scenes, so the ordering in the final enumeration will be "E D".
 *
 * <p>NAIVE_LINK_ORDER: a left-to-right preordering. In Starlark it is called "preorder"; its older
 * deprecated name is "naive_link".
 *
 * <p>For example, for the nested set {B, D, {A, C}}, the iteration order is "B D A C".
 *
 * <p>The order is called naive because it does no special treatment of dependency graphs that are
 * not trees. For such graphs the property of parent-before-dependencies in the iteration order will
 * not be upheld. For example, the diamond-shape graph A->{B, C}, B->{D}, C->{D} will be enumerated
 * as "A B D C" rather than "A B C D" or "A C B D".
 *
 * <p>The difference from LINK_ORDER is that this order gives priority to left-to-right order over
 * dependencies-after-parent ordering. Note that the latter is usually more important, so please use
 * LINK_ORDER whenever possible.
 */
// TODO(bazel-team): Remove deprecated names from the documentation above.
public enum Order {
  STABLE_ORDER("default"),
  COMPILE_ORDER("postorder"),
  LINK_ORDER("topological"),
  NAIVE_LINK_ORDER("preorder");

  private static final ImmutableMap<String, Order> VALUES;
  private static final Order[] ORDINALS;

  private final String starlarkName;
  private final NestedSet<?> emptySet;

  private Order(String starlarkName) {
    this.starlarkName = starlarkName;
    this.emptySet = new NestedSet<>(this);
  }

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Order STABLE_ORDER_CONSTANT = STABLE_ORDER;

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Order COMPILE_ORDER_CONSTANT = COMPILE_ORDER;

  @AutoCodec @AutoCodec.VisibleForSerialization static final Order LINK_ORDER_CONSTANT = LINK_ORDER;

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final Order NAIVE_LINK_ORDER_CONSTANT = NAIVE_LINK_ORDER;

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final NestedSet<?> EMPTY_STABLE = STABLE_ORDER.emptySet();

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final NestedSet<?> EMPTY_COMPILE = COMPILE_ORDER.emptySet();

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final NestedSet<?> EMPTY_LINK = LINK_ORDER.emptySet();

  @AutoCodec @AutoCodec.VisibleForSerialization
  static final NestedSet<?> EMPTY_NAIVE_LINK = NAIVE_LINK_ORDER.emptySet();

  /**
   * Returns an empty set of the given ordering.
   */
  @SuppressWarnings("unchecked")  // Nested sets are immutable, so a downcast is fine.
  <E> NestedSet<E> emptySet() {
    return (NestedSet<E>) emptySet;
  }

  public String getStarlarkName() {
    return starlarkName;
  }

  /**
   * Parses the given string as a nested set order
   *
   * @param name unique name of the order
   * @return the appropriate order instance
   * @throws IllegalArgumentException if the name is not valid
   */
  public static Order parse(String name) {
    if (VALUES.containsKey(name)) {
      return VALUES.get(name);
    } else {
      throw new IllegalArgumentException("Invalid order: " + name);
    }
  }

  /**
   * Determines whether two orders are considered compatible.
   *
   * <p>An order is compatible with itself (reflexivity) and all orders are compatible with
   * {@link #STABLE_ORDER}; the rest of the combinations are incompatible.
   */
  public boolean isCompatible(Order other) {
    return this == other || this == STABLE_ORDER || other == STABLE_ORDER;
  }

  /**
   * Indexes all possible values by name and stores the results in a {@code ImmutableMap}
   */
  static {
    ORDINALS = values();
    HashMap<String, Order> entries = Maps.newHashMapWithExpectedSize(ORDINALS.length);

    for (Order current : ORDINALS) {
      entries.put(current.getStarlarkName(), current);
    }

    VALUES = ImmutableMap.copyOf(entries);
  }

  static Order getOrder(int ordinal) {
    return ORDINALS[ordinal];
  }
}
