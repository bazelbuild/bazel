// Copyright 2025 The Bazel Authors. All rights reserved.
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
 * Interface for a {@code StarlarkValue} representing a reified type, i.e., a type that can be used
 * {@code isinstance()} check. In most cases, this is a {@link StarlarkCallable} which constructs
 * values of the associated type.
 */
public interface StarlarkTypeValue extends StarlarkValue {

  /**
   * Returns true if the given value is an instance of this value's associated type.
   *
   * <p>If {@code t.hasInstance(v)} is true in Java, then {@code isinstance(v, t)} is True in
   * Starlark. (The converse is not necessarily true, since there are other ways to register that a
   * Starlark value acts as a type.)
   */
  public boolean hasInstance(Object value);
}
