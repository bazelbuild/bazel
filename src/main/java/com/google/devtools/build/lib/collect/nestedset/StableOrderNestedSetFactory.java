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

import com.google.common.collect.ImmutableList;

/**
 * Stable order {@code NestedSet} factory.
 */
final class StableOrderNestedSetFactory implements NestedSetFactory {

  @Override
  public <E> NestedSet<E> onlyDirects(Object[] directs) {
    return new StableOnlyDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> onlyDirects(ImmutableList<E> directs) {
    return new StableImmutableListDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> oneDirectOneTransitive(E direct, NestedSet<E> transitive) {
    return new StableOneDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> manyDirectsOneTransitive(Object[] direct,
      NestedSet<E> transitive) {
    return new StableManyDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> onlyOneTransitive(NestedSet<E> transitive) {
    return new StableOnlyOneTransitiveNestedSet<>(transitive);
  }

  @Override
  public <E> NestedSet<E> onlyManyTransitives(NestedSet[] transitives) {
    return new StableOnlyTransitivesNestedSet<>(transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirectManyTransitive(Object direct, NestedSet[] transitives) {
    return new StableOneDirectManyTransitive<>(direct, transitives);
  }

  @Override
  public <E> NestedSet<E> manyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
    return new StableManyDirectManyTransitive<>(directs, transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirect(final E element) {
    return new StableSingleDirectNestedSet<>(element);
  }

  private static class StableOnlyDirectsNestedSet<E> extends OnlyDirectsNestedSet<E> {

    StableOnlyDirectsNestedSet(Object[] directs) { super(directs); }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableOneDirectOneTransitiveNestedSet<E> extends
      OneDirectOneTransitiveNestedSet<E> {

    private StableOneDirectOneTransitiveNestedSet(E direct, NestedSet<E> transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableOneDirectManyTransitive<E> extends OneDirectManyTransitive<E> {

    private StableOneDirectManyTransitive(Object direct, NestedSet[] transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableManyDirectManyTransitive<E> extends ManyDirectManyTransitive<E> {

    private StableManyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
      super(directs, transitives);
    }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableOnlyOneTransitiveNestedSet<E> extends OnlyOneTransitiveNestedSet<E> {

    private StableOnlyOneTransitiveNestedSet(NestedSet<E> transitive) { super(transitive); }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableManyDirectOneTransitiveNestedSet<E> extends
      ManyDirectOneTransitiveNestedSet<E> {

    private StableManyDirectOneTransitiveNestedSet(Object[] direct,
        NestedSet<E> transitive) { super(direct, transitive); }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableOnlyTransitivesNestedSet<E> extends OnlyTransitivesNestedSet<E> {

    private StableOnlyTransitivesNestedSet(NestedSet[] transitives) { super(transitives); }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }

  private static class StableImmutableListDirectsNestedSet<E> extends
      ImmutableListDirectsNestedSet<E> {

    private StableImmutableListDirectsNestedSet(ImmutableList<E> directs) { super(directs); }

    @Override
    public Order getOrder() {
      return Order.STABLE_ORDER;
    }
  }

  private static class StableSingleDirectNestedSet<E> extends SingleDirectNestedSet<E> {

    private StableSingleDirectNestedSet(E element) { super(element); }

    @Override
    public Order getOrder() { return Order.STABLE_ORDER; }
  }
}
