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

package com.google.devtools.build.lib.cmdline;

import com.google.devtools.build.lib.concurrent.ThreadSafety;

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

  public SymbolGenerator(T owner) {
    this.owner = owner;
  }

  @ThreadSafety.ThreadSafe
  public synchronized Symbol<T> generate() {
    return new Symbol<>(owner, index++);
  }

  private static final class Symbol<T> {
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
  };
}
