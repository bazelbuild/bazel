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

import javax.annotation.Nullable;

/**
 * A context for obtaining more detailed information about Starlark types.
 *
 * <p>This is used to inject type information from the {@code eval/} package into the {@code
 * syntax/} package, e.g. the method APIs of {@link StarlarkList}.
 */
public interface TypeContext {

  /**
   * Returns the type of the given field of a {@code list[T]} type, or null if no such field exists.
   */
  @Nullable
  StarlarkType getListFieldType(String name);

  /**
   * Returns the type of the given field of a {@code dict[K, V]} type, or null if no such field
   * exists.
   */
  @Nullable
  StarlarkType getDictFieldType(String name);

  /**
   * Returns the type of the given field of a {@code set[T]} type, or null if no such field exists.
   */
  @Nullable
  StarlarkType getSetFieldType(String name);
}
