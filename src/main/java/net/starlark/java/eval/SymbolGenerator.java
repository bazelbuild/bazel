// Copyright 2019 The Bazel Authors. All rights reserved.
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
package net.starlark.java.eval;

/**
 * Class to be used when an object wants to be compared using reference equality. Since reference
 * equality is not usable when comparing objects across multiple Starlark evaluations, we use a more
 * stable method: an object identifying the {@link #owner} of the current Starlark context, and an
 * {@link #index} indicating how many reference-equal objects have already been created (and
 * therefore asked for a unique symbol for themselves).
 *
 * <p>Objects that want to use reference equality should instead call {@link #generate} on a
 * provided {@code SymbolGenerator} instance, and compare the returned object for equality, since it
 * will be stable across identical Starlark evaluations.
 */
public final class SymbolGenerator<T> {
  private final T owner;
  private int index = 0;

  /**
   * Creates a new symbol generator for the Starlark evaluation uniquely identified by the given
   * owner.
   *
   * <p>Precisely, two {@code SymbolGenerators} that have owners {@code o1} and {@code o2} are
   * considered to be for the same Starlark evaluation, if and only if {@code o1.equals(o2)}.
   */
  public static <T> SymbolGenerator<T> create(T owner) {
    return new SymbolGenerator<>(owner);
  }

  /**
   * Creates a generator for a Starlark evaluation whose values don't require strict reference
   * equality checks.
   *
   * <p>This can be used in the following cases.
   *
   * <ul>
   *   <li>The result of a Starlark evaluation has a simple type (like numbers or strings) where
   *       values are compared, not object references.
   *   <li>The result is temporary and it won't be stored, transmitted or regenerated while being
   *       retained.
   * </ul>
   *
   * <p>The "regenerated while being retained" condition may occur, for example, if a part of the
   * resulting value is retained somewhere in the process, but the value itself is evicted from a
   * cache and is subsequently regenerated.
   */
  public static SymbolGenerator<Object> createTransient() {
    return create(new Object());
  }

  private SymbolGenerator(T owner) {
    this.owner = owner;
  }

  public synchronized Symbol<T> generate() {
    return new Symbol<>(owner, index++);
  }

  /** Identifier for an object created by a uniquely defined Starlark thread. */
  // TODO(bazel-team): The name "Symbol", in the context of an interpreter, is a bit confusing.
  // Consider renaming to "Token" or similar.
  public static final class Symbol<T> {
    private final T owner;
    private final int index;

    private Symbol(T owner, int index) {
      this.owner = owner;
      this.index = index;
    }

    @Override
    public boolean equals(Object o) {
      if (this == o) {
        return true;
      }
      if (!(o instanceof Symbol<?>)) {
        return false;
      }
      Symbol<?> symbol = (Symbol<?>) o;
      return index == symbol.index && owner.equals(symbol.owner);
    }

    @Override
    public int hashCode() {
      // We don't expect multiple indices for the same owner, save the computation.
      return owner.hashCode();
    }

    @Override
    public String toString() {
      return "<symbol=" + owner + ", index=" + index + ">";
    }
  }
}
