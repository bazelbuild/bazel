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
 * Type of a nested set (defines order). For explanation what these ordering mean,
 * see CompileOrderExpander, LinkOrderExpander, NaiveLinkOrderExpander.
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
