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
 * An annotation to mark built-in keyword argument methods accessible from Skylark.
 */
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkSignature {

  // TODO(bazel-team): parse most everything from single string specifying the signature
  // in Skylark syntax, e.g.: signature = "foo(a: string, b: ListOf(int)) -> NoneType"
  // String signature() default "";

  String name();

  String doc() default "";

  Param[] mandatoryPositionals() default {};

  Param[] optionalPositionals() default {};

  Param[] optionalNamedOnly() default {};

  Param[] mandatoryNamedOnly() default {};

  Param extraPositionals() default @Param(name = "");

  Param extraKeywords() default @Param(name = "");

  boolean documented() default true;

  Class<?> objectType() default Object.class;

  Class<?> returnType() default Object.class;

  // TODO(bazel-team): determine this way whether to accept mutable Lists
  // boolean mutableLists() default false;

  boolean useLocation() default false;

  boolean useAst() default false;

  boolean useEnvironment() default false;

  /**
   * An annotation for parameters of Skylark built-in functions.
   */
  @Retention(RetentionPolicy.RUNTIME)
  public @interface Param {

    String name();

    String doc() default "";

    String defaultValue() default "";

    Class<?> type() default Object.class;

    Class<?> generic1() default Object.class;

    boolean callbackEnabled() default false;

    boolean noneable() default false;

    // TODO(bazel-team): parse the type from a single field in Skylark syntax,
    // and allow a Union as "ThisType or ThatType or NoneType":
    // String type() default "Object";
  }
}
