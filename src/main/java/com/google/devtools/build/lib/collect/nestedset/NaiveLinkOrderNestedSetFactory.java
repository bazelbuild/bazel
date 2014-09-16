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

import com.google.common.collect.ImmutableList;

/**
 * NaiveLink order {@code NestedSet} factory.
 */
final class NaiveLinkOrderNestedSetFactory implements NestedSetFactory {

  @Override
  public <E> NestedSet<E> onlyDirects(Object[] directs) {
    return new NaiveLinkOnlyDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> onlyDirects(ImmutableList<E> directs) {
    return new NaiveLinkImmutableListDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> oneDirectOneTransitive(E direct, NestedSet<E> transitive) {
    return new NaiveLinkOneDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> manyDirectsOneTransitive(Object[] direct,
      NestedSet<E> transitive) {
    return new NaiveLinkManyDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> onlyOneTransitive(NestedSet<E> transitive) {
    return new NaiveLinkOnlyOneTransitiveNestedSet<>(transitive);
  }

  @Override
  public <E> NestedSet<E> onlyManyTransitives(NestedSet[] transitives) {
    return new NaiveLinkOnlyTransitivesNestedSet<>(transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirectManyTransitive(Object direct, NestedSet[] transitives) {
    return new NaiveLinkOneDirectManyTransitive<>(direct, transitives);
  }

  @Override
  public <E> NestedSet<E> manyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
    return new NaiveLinkManyDirectManyTransitive<>(directs, transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirect(final E element) {
    return new NaiveLinkSingleDirectNestedSet<>(element);
  }

  private static class NaiveLinkOnlyDirectsNestedSet<E> extends OnlyDirectsNestedSet<E> {

    NaiveLinkOnlyDirectsNestedSet(Object[] directs) { super(directs); }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkOneDirectOneTransitiveNestedSet<E> extends
      OneDirectOneTransitiveNestedSet<E> {

    private NaiveLinkOneDirectOneTransitiveNestedSet(E direct, NestedSet<E> transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkOneDirectManyTransitive<E> extends OneDirectManyTransitive<E> {

    private NaiveLinkOneDirectManyTransitive(Object direct, NestedSet[] transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkManyDirectManyTransitive<E> extends ManyDirectManyTransitive<E> {

    private NaiveLinkManyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
      super(directs, transitives);
    }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkOnlyOneTransitiveNestedSet<E>
      extends OnlyOneTransitiveNestedSet<E> {

    private NaiveLinkOnlyOneTransitiveNestedSet(NestedSet<E> transitive) { super(transitive); }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkManyDirectOneTransitiveNestedSet<E> extends
      ManyDirectOneTransitiveNestedSet<E> {

    private NaiveLinkManyDirectOneTransitiveNestedSet(Object[] direct,
        NestedSet<E> transitive) { super(direct, transitive); }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkOnlyTransitivesNestedSet<E> extends OnlyTransitivesNestedSet<E> {

    private NaiveLinkOnlyTransitivesNestedSet(NestedSet[] transitives) { super(transitives); }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }

  private static class NaiveLinkImmutableListDirectsNestedSet<E> extends
      ImmutableListDirectsNestedSet<E> {

    private NaiveLinkImmutableListDirectsNestedSet(ImmutableList<E> directs) { super(directs); }

    @Override
    public Order getOrder() {
      return Order.NAIVE_LINK_ORDER;
    }
  }

  private static class NaiveLinkSingleDirectNestedSet<E> extends SingleDirectNestedSet<E> {

    private NaiveLinkSingleDirectNestedSet(E element) { super(element); }

    @Override
    public Order getOrder() { return Order.NAIVE_LINK_ORDER; }
  }
}
