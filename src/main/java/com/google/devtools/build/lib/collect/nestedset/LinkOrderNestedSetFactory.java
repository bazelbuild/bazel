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
 * Link order {@code NestedSet} factory.
 */
final class LinkOrderNestedSetFactory implements NestedSetFactory {

  @Override
  public <E> NestedSet<E> onlyDirects(Object[] directs) {
    return new LinkOnlyDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> onlyDirects(ImmutableList<E> directs) {
    return new LinkImmutableListDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> oneDirectOneTransitive(E direct, NestedSet<E> transitive) {
    return new LinkOneDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> manyDirectsOneTransitive(Object[] direct,
      NestedSet<E> transitive) {
    return new LinkManyDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> onlyOneTransitive(NestedSet<E> transitive) {
    return new LinkOnlyOneTransitiveNestedSet<>(transitive);
  }

  @Override
  public <E> NestedSet<E> onlyManyTransitives(NestedSet[] transitives) {
    return new LinkOnlyTransitivesNestedSet<>(transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirectManyTransitive(Object direct, NestedSet[] transitives) {
    return new LinkOneDirectManyTransitive<>(direct, transitives);
  }

  @Override
  public <E> NestedSet<E> manyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
    return new LinkManyDirectManyTransitive<>(directs, transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirect(E element) {
    return new LinkSingleDirectNestedSet<>(element);
  }

  private static class LinkOnlyDirectsNestedSet<E> extends OnlyDirectsNestedSet<E> {

    LinkOnlyDirectsNestedSet(Object[] directs) { super(directs); }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkOneDirectOneTransitiveNestedSet<E> extends
      OneDirectOneTransitiveNestedSet<E> {

    private LinkOneDirectOneTransitiveNestedSet(E direct, NestedSet<E> transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkOneDirectManyTransitive<E> extends OneDirectManyTransitive<E> {

    private LinkOneDirectManyTransitive(Object direct, NestedSet[] transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkManyDirectManyTransitive<E> extends ManyDirectManyTransitive<E> {

    private LinkManyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
      super(directs, transitives);
    }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkOnlyOneTransitiveNestedSet<E> extends OnlyOneTransitiveNestedSet<E> {

    private LinkOnlyOneTransitiveNestedSet(NestedSet<E> transitive) { super(transitive); }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkManyDirectOneTransitiveNestedSet<E> extends
      ManyDirectOneTransitiveNestedSet<E> {

    private LinkManyDirectOneTransitiveNestedSet(Object[] direct,
        NestedSet<E> transitive) { super(direct, transitive); }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkOnlyTransitivesNestedSet<E> extends OnlyTransitivesNestedSet<E> {

    private LinkOnlyTransitivesNestedSet(NestedSet[] transitives) { super(transitives); }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }

  private static class LinkImmutableListDirectsNestedSet<E> extends
      ImmutableListDirectsNestedSet<E> {

    private LinkImmutableListDirectsNestedSet(ImmutableList<E> directs) { super(directs); }

    @Override
    public Order getOrder() {
      return Order.LINK_ORDER;
    }
  }

  private static class LinkSingleDirectNestedSet<E> extends SingleDirectNestedSet<E> {

    private LinkSingleDirectNestedSet(E element) { super(element); }

    @Override
    public Order getOrder() { return Order.LINK_ORDER; }
  }
}
