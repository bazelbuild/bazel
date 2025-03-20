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

import com.google.auto.value.AutoValue;

/**
 * Class to be used when an object wants to be compared using reference equality. Since reference
 * equality is not usable when comparing objects across multiple Starlark evaluations, we use a more
 * stable method: an object identifying the {@link #owner} of the current Starlark context, and an
 * {@link #index} indicating how many reference-equal objects have already been created (and
 * therefore asked for a unique symbol for themselves). Global symbols may also be identified using
 * their exported name rather than an anonymous index.
 *
 * <p>Objects that want to use reference equality should instead call {@link #generate} on a
 * provided {@code SymbolGenerator} instance, and compare the returned object for equality, since it
 * will be stable across identical Starlark evaluations. Note that equality comparisons are
 * invalidated by any change to the inputs of a Starlark evaluation. For example, it is not valid to
 * compare two values that came from different Bazel builds with an intervening edit to a .bzl file.
 *
 * <p>For Starlark values that rely on this class, equality comparison across Starlark threads is
 * not guaranteed to be consistent until both threads are done running. This is due to the edge case
 * of one value being exported while the other is still unexported, since the export process can
 * change the equality token.
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
    return LocalSymbol.create(owner, index++);
  }

  T getOwner() {
    return owner;
  }

  /** Identifier for an object created by a uniquely defined Starlark thread. */
  // TODO(bazel-team): The name "Symbol", in the context of an interpreter, is a bit confusing.
  // Consider renaming to "Token" or similar.
  public abstract static class Symbol<T> {
    /**
     * Creates a new {@link GlobalSymbol} with the same owner as this symbol.
     *
     * <p>Objects may start with a {@link LocalSymbol} and are later exported with a global name.
     * This method can be used to create a suitable {@link GlobalSymbol}.
     */
    public final GlobalSymbol<T> exportAs(String name) {
      return GlobalSymbol.create(getOwner(), name);
    }

    public abstract T getOwner();

    public abstract boolean isGlobal();
  }

  @AutoValue
  abstract static class LocalSymbol<T> extends Symbol<T> {
    private static <T> LocalSymbol<T> create(T owner, int index) {
      return new AutoValue_SymbolGenerator_LocalSymbol<>(owner, index);
    }

    abstract int getIndex();

    @Override
    public final boolean isGlobal() {
      return false;
    }
  }

  /**
   * An identifier for a global variable.
   *
   * <p>Intended as an optimization, allowing the lookup of a global variable from its GlobalSymbol,
   * e.g. for deserialization: the owner should be a wrapper object for a {@link Module}, and we can
   * obtain the value from the symbol's name and {@link Module#getGlobal}.
   */
  @AutoValue
  public abstract static class GlobalSymbol<T> extends Symbol<T> {
    private static <T> GlobalSymbol<T> create(T owner, String name) {
      return new AutoValue_SymbolGenerator_GlobalSymbol<>(owner, name);
    }

    public abstract String getName();

    @Override
    public final boolean isGlobal() {
      return true;
    }
  }
}
