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
import com.google.common.collect.ImmutableSet;
import java.util.Collection;
import java.util.List;
import javax.annotation.Nullable;
import net.starlark.java.syntax.Resolver.Module;
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
   * A basic static resolver Module implementation for testing.
   *
   * <p>It defines only the given predeclared names, without even universals (e.g. None). No type
   * constructors are resolved.
   */
  public static class ModuleWithPredeclared implements Module {
    private final ImmutableSet<String> names;

    public ModuleWithPredeclared(Collection<String> names) {
      this.names = ImmutableSet.copyOf(names);
    }

    public ModuleWithPredeclared(String... names) {
      this(ImmutableSet.copyOf(names));
    }

    @Override
    public Scope resolve(String name) throws Undefined {
      if (names.contains(name)) {
        return Scope.PREDECLARED;
      } else {
        throw new Undefined(String.format("name '%s' is not defined", name), names);
      }
    }

    @Override
    @Nullable
    public TypeConstructor getTypeConstructor(String name) throws Undefined {
      throw new Undefined("ModuleWithPredeclared does not support type resolution");
    }
  }

  /** A version of {@link ModuleWithPredeclared} that knows about the universal types. */
  public static class ModuleWithUniversalTypes extends ModuleWithPredeclared {
    public ModuleWithUniversalTypes() {
      super(Types.TYPE_UNIVERSE.keySet());
    }

    @Override
    @Nullable
    public TypeConstructor getTypeConstructor(String name) throws Undefined {
      resolve(name); // throws if unknown
      return Types.TYPE_UNIVERSE.get(name);
    }
  }
}
