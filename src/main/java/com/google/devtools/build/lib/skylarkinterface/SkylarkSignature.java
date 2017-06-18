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
 *
 * <p>Use this annotation around a {@link com.google.devtools.build.lib.syntax.BuiltinFunction} or
 * a {@link com.google.devtools.build.lib.syntax.BuiltinFunction.Factory}. The annotated function
 * should expect the arguments described by {@link #parameters()}, {@link #extraPositionals()},
 * and {@link #extraKeywords()}. It should also expect the following extraneous arguments:
 *
 * <ul>
 *   <li>
 *     {@link com.google.devtools.build.lib.events.Location} if {@link #useLocation()} is
 *     true.
 *   </li>
 *   <li>{@link com.google.devtools.build.lib.syntax.ASTNode} if {@link #useAst()} is true.</li>
 *   <li>
 *     {@link com.google.devtools.build.lib.syntax.Environment} if {@link #useEnvironment()} )}
 *     is true.
 *   </li>
 * </ul>
 */
@Target({ElementType.FIELD})
@Retention(RetentionPolicy.RUNTIME)
public @interface SkylarkSignature {

  // TODO(bazel-team): parse most everything from single string specifying the signature
  // in Skylark syntax, e.g.: signature = "foo(a: string, b: ListOf(int)) -> NoneType"
  // String signature() default "";

  /**
   * Name of the method as exposed to Skylark.
   */
  String name();

  /**
   * General documentation block of the method. See the skylark documentation at
   * http://www.bazel.build/docs/skylark/.
   */
  String doc() default "";

  /**
   * List of parameters for calling this method. Named only parameters are expected to be last.
   */
  Param[] parameters() default {};

  /**
   * Defines a catch all positional parameters. By default, it is an error to define more
   * positional parameters that specified but by defining an extraPositionals argument, one can
   * catch those. See python's <code>*args</code>
   * (http://thepythonguru.com/python-args-and-kwargs/).
   */
  Param extraPositionals() default @Param(name = "");

  /**
   * Defines a catch all named parameters. By default, it is an error to define more
   * named parameters that specified but by defining an extraKeywords argument, one can catch those.
   * See python's <code>**kwargs</code> (http://thepythonguru.com/python-args-and-kwargs/).
   */
  Param extraKeywords() default @Param(name = "");

  /**
   * Set <code>documented</code> to <code>false</code> if this method should not be mentioned
   * in the documentation of Skylark. This is generally used for experimental APIs or duplicate
   * methods already documented on another call.
   */
  boolean documented() default true;

  /**
   * Type of the object associated to that function. If this field is
   * <code>Object.class</code>, then the function will be considered as an object method.
   * For example, to add a function to the string object, set it to <code>String.class</code>.
   */
  Class<?> objectType() default Object.class;

  /**
   * Return type of the function. Use {@link com.google.devtools.build.lib.syntax.Runtime.NoneType}
   * for a void function.
   */
  Class<?> returnType() default Object.class;

  /**
   * Fake return type of the function. Used by the documentation generator for documenting
   * deprecated functions (documentation for this type is generated, even if it's not the real
   * return type).
   */
  Class<?> documentationReturnType() default Object.class;

  // TODO(bazel-team): determine this way whether to accept mutable Lists
  // boolean mutableLists() default false;

  /**
   * If true the location of the call site will be passed as an argument of the annotated function.
   */
  boolean useLocation() default false;

  /**
   * If true the AST of the call site will be passed as an argument of the annotated function.
   */
  boolean useAst() default false;

  /**
   * If true the AST of the Skylark Environment
   * ({@link com.google.devtools.build.lib.syntax.Environment}) will be passed as an argument of the
   * annotated function.
   */
  boolean useEnvironment() default false;
}
