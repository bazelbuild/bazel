// Copyright 2014 The Bazel Authors. All rights reserved.
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

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

import javax.annotation.Nullable;

/**
 * An annotation to mark Skylark modules or Skylark accessible Java data types.
 * A Skylark modules always corresponds to exactly one Java class.
 */
@Target({ElementType.TYPE})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkModule {

  String name();

  String doc();

  boolean documented() default true;

  boolean namespace() default false;

  /** Helper method to quickly get the SkylarkModule name of a class (if present). */
  public static final class Resolver {
    /**
     * Returns the Skylark name of the given class or null, if the SkylarkModule annotation is not
     * present.
     */
    @Nullable
    public static String resolveName(Class<?> clazz) {
      SkylarkModule annotation = clazz.getAnnotation(SkylarkModule.class);
      return (annotation == null) ? null : annotation.name();
    }

    /** Utility method only. */
    private Resolver() {}
  }
}
