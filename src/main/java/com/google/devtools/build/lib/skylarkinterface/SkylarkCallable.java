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

/**
 * A marker interface for Java methods which can be called from Skylark.
 */
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkCallable {

  /**
   * Name of the method, as exposed to Skylark.
   */
  String name() default "";

  /**
   * The documentation text in Skylark. It can contain HTML tags for special formatting.
   *
   * <p>It is allowed to be empty only if {@link #documented()} is false.
   */
  String doc() default "";

  /**
   * If true, the function will appear in the Skylark documentation. Set this to false if the
   * function is experimental or an overloading and doesn't need to be documented.
   */
  boolean documented() default true;

  /**
   * If true, this method will be considered as a field of the enclosing Java object. E.g., if set
   * to true on a method {@code foo}, then the callsites of this method will look like
   * {@code bar.foo} instead of {@code bar.foo()}. The annotated method must be parameterless.
   */
  boolean structField() default false;

  /**
   * Set it to true if the Java method may return <code>null</code> (which will then be converted to
   * <code>None</code>). If not set and the Java method returns null, an error will be raised.
   */
  boolean allowReturnNones() default false;
}
