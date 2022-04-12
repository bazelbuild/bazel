// Copyright 2016 The Bazel Authors. All rights reserved.
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
package net.starlark.java.annot;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/** An annotation for parameter types for Starlark built-in functions. */
@Retention(RetentionPolicy.RUNTIME)
public @interface ParamType {
  /**
   * The Java class of the type, e.g. {@link String}.class or {@link
   * net.starlark.java.eval.Sequence}.class.
   */
  Class<?> type();

  /**
   * When {@link #type()} is a generic type (e.g., {@link net.starlark.java.eval.Sequence}), specify
   * the type parameter (e.g. {@link String}.class} along with {@link
   * net.starlark.java.eval.Sequence} for {@link #type()} to specify a list of strings).
   *
   * <p>This is only used for documentation generation. The actual generic type is not checked at
   * runtime, so the Java method signature should use a generic type of Object and cast
   * appropriately.
   */
  // TODO(#13365): make this a data structure so we can represent a {@link
  // net.starlark.java.eval.Sequence} of types {@code A} or {@code B} intermixed, a {@link
  // net.starlark.java.eval.Dict} mapping from {@code A} to {@code B}, etc.
  Class<?> generic1() default Object.class;
}
