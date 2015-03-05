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

/**
 * Type of a nested set (defines order).
 */
public enum Order {

  STABLE_ORDER(new CompileOrderExpander<>(), new StableOrderNestedSetFactory()),
  COMPILE_ORDER(new CompileOrderExpander<>(), new CompileOrderNestedSetFactory()),
  LINK_ORDER(new LinkOrderExpander<>(), new LinkOrderNestedSetFactory()),
  NAIVE_LINK_ORDER(new NaiveLinkOrderExpander<>(), new NaiveLinkOrderNestedSetFactory());

  private final NestedSetExpander<?> expander;
  final NestedSetFactory factory;
  private final NestedSet<?> emptySet;

  private Order(NestedSetExpander<?> expander, NestedSetFactory factory) {
    this.expander = expander;
    this.factory = factory;
    this.emptySet = new EmptyNestedSet<>(this);
  }

  /**
   * Returns an empty set of the given ordering.
   */
  @SuppressWarnings("unchecked")  // Nested sets are immutable, so a downcast is fine.
  <E> NestedSet<E> emptySet() {
    return (NestedSet<E>) emptySet;
  }

  /**
   * Returns an empty set of the given ordering.
   */
  @SuppressWarnings("unchecked")  // Nested set expanders contain no data themselves.
  <E> NestedSetExpander<E> expander() {
    return (NestedSetExpander<E>) expander;
  }

}
