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
package com.google.devtools.build.lib.skylarkinterface;

import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;

/**
 * An annotation for parameter types for Skylark built-in functions.
 */
@Retention(RetentionPolicy.RUNTIME)
public @interface ParamType {
  /**
   * The Java class of the type, e.g. {@link String}.class or
   * {@link com.google.devtools.build.lib.syntax.SkylarkList}.class.
   */
  Class<?> type() default Object.class;

  /**
   * When {@link #type()} is a generic type (e.g.,
   * {@link com.google.devtools.build.lib.syntax.SkylarkList}), specify the type parameter (e.g.
   * {@link String}.class} along with {@link com.google.devtools.build.lib.syntax.SkylarkList} for
   * {@link #type()} to specify a list of strings).
   */
  Class<?> generic1() default Object.class;
}
