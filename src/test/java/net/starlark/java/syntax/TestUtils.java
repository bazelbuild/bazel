// Copyright 2026 The Bazel Authors. All rights reserved.
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

package net.starlark.java.syntax;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.Set;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver.Module.Undefined;
import net.starlark.java.syntax.Resolver.Scope;

public final class TestUtils {

  private TestUtils() {}

  /**
   * Returns the first error whose string form contains the specified substring, or throws an
   * informative AssertionError if there is none.
   */
  public static SyntaxError assertContainsError(List<SyntaxError> errors, String substr) {
    for (SyntaxError error : errors) {
      if (error.toString().contains(substr)) {
        return error;
      }
    }
    if (errors.isEmpty()) {
      throw new AssertionError("no errors, want '" + substr + "'");
    } else {
      throw new AssertionError(
          "error '" + substr + "' not found, but got these:\n" + Joiner.on("\n").join(errors));
    }
  }

  /**
   * A static resolver {@link net.starlark.java.syntax.Resolver.Module} implementation, for tests of
   * the resolver and type checker.
   *
   * <p>This {@code Module} only supports predeclared symbols, not universals or predefined globals.
   * The absence of universals means that any test cases relying on this {@code Module} cannot
   * process Starlark code snippets containing builtin singletons ({@code None}/{@code True}/{@code
   * False}) or functions ({@code len()}, etc.).
   */
  public static class Module implements net.starlark.java.syntax.Resolver.Module {
    private final ImmutableSet<String> predeclared;
    private final ImmutableMap<String, TypeConstructor> typeConstructors;

    private Module(Set<String> predeclared, Map<String, TypeConstructor> typeConstructors) {
      this.predeclared = ImmutableSet.copyOf(predeclared);
      this.typeConstructors = ImmutableMap.copyOf(typeConstructors);
    }

    /**
     * Returns a Module with the given names as predeclared symbols, which are not type
     * constructors.
     */
    public static Module withPredeclared(String... names) {
      return withPredeclared(ImmutableSet.copyOf(names));
    }

    /**
     * Returns a Module with the given names as predeclared symbols, which are not type
     * constructors.
     */
    public static Module withPredeclared(Collection<String> names) {
      return new Module(ImmutableSet.copyOf(names), ImmutableMap.of());
    }

    /**
     * Returns a Module with the given predeclared names and (optional) associated type
     * constructors.
     *
     * <p>Arguments must be alternating pairs of {@code String}s and {@link TypeConstructor}s. The
     * {@code TypeConstructor} references may be null, which indicates that the corresponding name
     * cannot be used as a type constructor. For example, {@code withTypes("foo", null)} is
     * equivalent to {@code withPredeclared("foo")}.
     */
    public static Module withTypes(Object... args) {
      return withTypes(ImmutableMap.of(), args);
    }

    /**
     * Same as {@link #withTypes(Object...)}, but accepts entries specified via {@code base} in
     * addition to the alternating pairs given in {@code args}.
     */
    public static Module withTypes(Map<String, TypeConstructor> base, Object... args) {
      ImmutableSet.Builder<String> predeclared = ImmutableSet.builder();
      ImmutableMap.Builder<String, TypeConstructor> typeConstructors = ImmutableMap.builder();
      for (Map.Entry<String, TypeConstructor> entry : base.entrySet()) {
        predeclared.add(entry.getKey());
        if (entry.getValue() != null) {
          typeConstructors.put(entry);
        }
      }
      Preconditions.checkArgument(args.length % 2 == 0, "`args` must have an even length");
      for (int i = 0; i < args.length; i += 2) {
        String name = (String) args[i];
        TypeConstructor tc = (TypeConstructor) args[i + 1];
        predeclared.add(name);
        if (tc != null) {
          typeConstructors.put(name, tc);
        }
      }
      return new Module(predeclared.build(), typeConstructors.buildOrThrow());
    }

    /** Returns a Module with the universal type constructors. */
    public static Module withUniversalTypes() {
      return withTypes(Types.TYPE_UNIVERSE);
    }

    /** Same as {@link #withTypes(Object...)}, but includes the universal types. */
    public static Module withUniversalTypesAnd(Object... args) {
      return withTypes(Types.TYPE_UNIVERSE, args);
    }

    @Override
    public Scope resolve(String name) throws Undefined {
      if (predeclared.contains(name)) {
        return Scope.PREDECLARED;
      } else {
        throw new Undefined(String.format("name '%s' is not defined", name), predeclared);
      }
    }

    @Override
    @Nullable
    public TypeConstructor getTypeConstructor(String name) throws Undefined {
      resolve(name); // throws if unknown
      return typeConstructors.get(name);
    }
  }
}
