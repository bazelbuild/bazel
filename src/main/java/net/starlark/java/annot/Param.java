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

/** An annotation for parameters of Starlark built-in functions. */
@Retention(RetentionPolicy.RUNTIME)
public @interface Param {

  /**
   * Name of the parameter, as viewed from Starlark. Used for matching keyword arguments and for
   * generating documentation.
   */
  String name();

  /**
   * Documentation of the parameter.
   */
  String doc() default "";

  /**
   * Determines whether the parameter appears in generated documentation. Set this to false to
   * suppress parameters whose use is intentionally restricted.
   *
   * <p>An undocumented parameter must be {@link #named} and may not be followed by positional
   * parameters or {@code **kwargs}.
   */
  boolean documented() default true;

  /**
   * Default value for the parameter, written as a Starlark expression (e.g. "False", "True", "[]",
   * "None").
   *
   * <p>If this is empty (the default), the parameter is treated as mandatory. (Thus an exception
   * will be thrown if left unspecified by the caller).
   *
   * <p>If the function implementation needs to distinguish the case where the caller does not
   * supply a value for this parameter, you can set the default to the magic string "unbound", which
   * maps to the sentinal object {@link net.starlark.java.eval.Starlark#UNBOUND} (which can't appear
   * in normal Starlark code).
   */
  String defaultValue() default "";

  /**
   * List of allowed types for the parameter.
   *
   * <p>The array may be omitted, in which case the parameter accepts any value whose class is
   * assignable to the class of the parameter variable.
   *
   * <p>If a function should accept None, NoneType should be in this list.
   */
  ParamType[] allowedTypes() default {};

  /**
   * If true, the parameter may be specified as a named parameter. For example for an integer named
   * parameter {@code foo} of a method {@code bar}, then the method call will look like {@code
   * bar(foo=1)}.
   *
   * <p>If false, then {@link #positional} must be true (otherwise there is no way to reference the
   * parameter via an argument).
   *
   * <p>If this parameter represents the 'extra positionals' (args) or 'extra keywords' (kwargs)
   * element of a method, this field has no effect.
   */
  boolean named() default false;

  /**
   * If true, the parameter may be specified as a positional parameter. For example for an integer
   * positional parameter {@code foo} of a method {@code bar}, then the method call will look like
   * {@code bar(1)}. If {@link #named()} is {@code false}, then this will be the only way to call
   * {@code bar}.
   *
   * <p>If false, then {@link #named} must be true (otherwise there is no way to reference the
   * parameter via an argument)
   *
   * <p>Positional arguments should come first.
   *
   * <p>If this parameter represents the 'extra positionals' (args) or 'extra keywords' (kwargs)
   * element of a method, this field has no effect.
   */
  boolean positional() default true;

  /**
   * If non-empty, the annotated parameter will only be present if the given semantic flag is true.
   * (If the parameter is disabled, it may not be specified by a user, and the Java method will
   * always be invoked with the parameter set to its default value.)
   *
   * <p>Note that at most one of {@link #enableOnlyWithFlag} and {@link #disableWithFlag} can be
   * non-empty.
   *
   * <p>If {@link #enableOnlyWithFlag} is non-empty, then {@link #defaultValue} must also be
   * non-empty; mandatory parameters cannot be toggled by a flag.
   */
  String enableOnlyWithFlag() default "";

  /**
   * If non-empty, the annotated parameter will only be present if the given semantic flag is false.
   * (If the parameter is disabled, it may not be specified by a user, and the Java method will
   * always be invoked with the parameter set to its default value.)
   *
   * <p>Note that at most one of {@link #enableOnlyWithFlag} and {@link #disableWithFlag} can be
   * non-empty.
   *
   * <p>If {@link #disableWithFlag} is non-empty, then {@link #defaultValue} must also be non-empty;
   * mandatory parameters cannot be toggled by a flag.
   */
  String disableWithFlag() default "";
}
