// Copyright 2018 The Bazel Authors. All rights reserved.
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
 * An annotation to mark {@link SkylarkCallable}-annotated methods as representing top-level
 * constructors for other Skylark objects. This is used only for documentation purposes.
 *
 * <p>For example, a "Foo" type skylark object might be constructable at the top level using
 * a global callable "Foo()". One can annotate that callable with this annotation to ensure that
 * the documentation for "Foo()" appears alongside the documentation for the Foo type, and not
 * the available globals.
 */
@Target({ElementType.METHOD})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkConstructor {

  /**
   * The java class of the skylark type that this annotation's method is a constructor for.
   */
  Class<?> objectType();

  /**
   * If non-empty, documents the way to invoke this function from the top level.
   *
   * <p>For example, if this constructs objects of type Foo, and this value is set to "Bar.Baz",
   * then the documentation for the method signature will appear as "Foo Bar.Baz(...)".
   */
  String receiverNameForDoc() default "";
}
