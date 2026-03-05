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

package net.starlark.java.lib;

import net.starlark.java.eval.StarlarkSemantics;

/** An interface allowing Starlark values to define their own structured text encoding. */
public interface StarlarkEncodable {
  /**
   * Returns a value which represents this object and which will be encoded by {@link
   * net.starlark.java.lib.json.Json#encode} and {@link
   * com.google.devtools.build.lib.packages.Proto.TextEncoder}.
   *
   * <p>The returned value must be one of the following:
   *
   * <ul>
   *   <li>{@link net.starlark.java.eval.Starlark#NONE}
   *   <li>a {@link Boolean}, {@link String}, {@link net.starlark.java.eval.StarlarkInt}, or {@link
   *       net.starlark.java.eval.StarlarkFloat}
   *   <li>a {@link java.util.Map} (for example, a {@link net.starlark.java.eval.Dict}). For
   *       compatibility with all encoders, the keys must be strings, and the values must be
   *       encodable scalars or structs.
   *   <li>a {@link net.starlark.java.eval.StarlarkIterable} of encodable elements. For
   *       compatibility with all encoders, the elements must be encodable scalars or structs.
   *   <li>a {@link net.starlark.java.eval.Structure} with encodable field values.
   * </ul>
   *
   * <p>Returning a {@code Structure} is recommended, unless there is a strong reason otherwise.
   */
  Object objectForEncoding(StarlarkSemantics semantics);
}
