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
import java.util.List;
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

  private static class TestModule implements Module {
    private final ImmutableSet<String> predeclared;

    TestModule(ImmutableSet<String> predeclared) {
      this.predeclared = predeclared;
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
    public Object resolveType(String name) throws Undefined {
      throw new Undefined("TestModule does not support type resolution");
    }
  }

  /**
   * A basic static resolver Module implementation for testing.
   *
   * <p>It defines only the given predeclared names, without even universals (e.g. None). No types
   * are resolved.
   */
  public static Module moduleWithPredeclared(String... names) {
    return new TestModule(ImmutableSet.copyOf(names));
  }

  /** A basic static resolver Module implementation that knows about the universal types. */
  public static Module moduleWithUniversalTypes() {
    return new TestModule(Types.TYPE_UNIVERSE.keySet()) {
      @Override
      public Object resolveType(String name) throws Undefined {
        resolve(name); // throws if unknown
        Object type = Types.TYPE_UNIVERSE.get(name);
        // TODO: #28043 - Remove this assertion when we simplify the universe schema.
        if (!(type instanceof StarlarkType || type instanceof Types.TypeConstructorProxy)) {
          throw new AssertionError("invalid type universe");
        }
        return type;
      }
    };
  }
}
