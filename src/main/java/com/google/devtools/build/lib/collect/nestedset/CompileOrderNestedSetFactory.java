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
 * Compile order {@code NestedSet} factory.
 */
final class CompileOrderNestedSetFactory implements NestedSetFactory {

  @Override
  public <E> NestedSet<E> onlyDirects(Object[] directs) {
    return new CompileOnlyDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> onlyDirects(ImmutableList<E> directs) {
    return new CompileOrderImmutableListDirectsNestedSet<>(directs);
  }

  @Override
  public <E> NestedSet<E> oneDirectOneTransitive(E direct, NestedSet<E> transitive) {
    return new CompileOneDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> manyDirectsOneTransitive(Object[] direct,
      NestedSet<E> transitive) {
    return new CompileManyDirectOneTransitiveNestedSet<>(direct, transitive);
  }

  @Override
  public <E> NestedSet<E> onlyOneTransitive(NestedSet<E> transitive) {
    return new CompileOnlyOneTransitiveNestedSet<>(transitive);
  }

  @Override
  public <E> NestedSet<E> onlyManyTransitives(NestedSet[] transitives) {
    return new CompileOnlyTransitivesNestedSet<>(transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirectManyTransitive(Object direct, NestedSet[] transitives) {
    return new CompileOneDirectManyTransitive<>(direct, transitives);
  }

  @Override
  public <E> NestedSet<E> manyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
    return new CompileManyDirectManyTransitive<>(directs, transitives);
  }

  @Override
  public <E> NestedSet<E> oneDirect(E element) {
    return new CompileSingleDirectNestedSet<>(element);
  }

  private static class CompileOnlyDirectsNestedSet<E> extends OnlyDirectsNestedSet<E> {

    CompileOnlyDirectsNestedSet(Object[] directs) { super(directs); }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileOneDirectOneTransitiveNestedSet<E> extends
      OneDirectOneTransitiveNestedSet<E> {

    private CompileOneDirectOneTransitiveNestedSet(E direct, NestedSet<E> transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileOneDirectManyTransitive<E> extends OneDirectManyTransitive<E> {

    private CompileOneDirectManyTransitive(Object direct, NestedSet[] transitive) {
      super(direct, transitive);
    }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileManyDirectManyTransitive<E> extends ManyDirectManyTransitive<E> {

    private CompileManyDirectManyTransitive(Object[] directs, NestedSet[] transitives) {
      super(directs, transitives);
    }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileOnlyOneTransitiveNestedSet<E> extends OnlyOneTransitiveNestedSet<E> {

    private CompileOnlyOneTransitiveNestedSet(NestedSet<E> transitive) { super(transitive); }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileManyDirectOneTransitiveNestedSet<E> extends
      ManyDirectOneTransitiveNestedSet<E> {

    private CompileManyDirectOneTransitiveNestedSet(Object[] direct,
        NestedSet<E> transitive) { super(direct, transitive); }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileOnlyTransitivesNestedSet<E> extends OnlyTransitivesNestedSet<E> {

    private CompileOnlyTransitivesNestedSet(NestedSet[] transitives) { super(transitives); }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }

  private static class CompileOrderImmutableListDirectsNestedSet<E> extends
      ImmutableListDirectsNestedSet<E> {

    private CompileOrderImmutableListDirectsNestedSet(ImmutableList<E> directs) { super(directs); }

    @Override
    public Order getOrder() {
      return Order.COMPILE_ORDER;
    }
  }

  private static class CompileSingleDirectNestedSet<E> extends SingleDirectNestedSet<E> {

    private CompileSingleDirectNestedSet(E element) { super(element); }

    @Override
    public Order getOrder() { return Order.COMPILE_ORDER; }
  }
}
