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
   * Type of the parameter, e.g. {@link String}.class or {@link
   * net.starlark.java.eval.Sequence}.class.
   */
  Class<?> type() default Object.class;

  /**
   * List of allowed types for the parameter if multiple types are allowed.
   *
   * <p>If using this, {@link #type()} should be set to {@code Object.class}.
   */
  ParamType[] allowedTypes() default {};

  /**
   * When {@link #type()} is a generic type (e.g., {@link net.starlark.java.eval.Sequence}), specify
   * the type parameter (e.g. {@link String}.class} along with {@link
   * net.starlark.java.eval.Sequence} for {@link #type()} to specify a list of strings).
   *
   * <p>This is only used for documentation generation. The actual generic type is not checked at
   * runtime, so the Java method signature should use a generic type of Object and cast
   * appropriately.
   */
  Class<?> generic1() default Object.class;

  /**
   * Indicates whether this parameter accepts {@code None} as a value, its allowed types
   * notwithstanding.
   *
   * <p>If true, {@code None} is accepted as a valid input in addition to the types mentioned by
   * {@link #type} or {@link #allowedTypes}. In this case, the Java type of the corresponding method
   * parameter must be {@code Object}.
   *
   * <p>If false, this parameter cannot be passed {@code None}, even if it would otherwise be
   * allowed by {@code type} or {@code allowedTypes}.
   */
  // TODO(starlark-team): Allow None as a value when noneable is false and the type is Object. But
  // look out for unwanted user-visible changes in the signatures of builtins.
  // TODO(140932420): Consider simplifying noneable by converting None to null, so that the Java
  // type need not be Object. But note that we still have the same problem for params whose default
  // value is the special "unbound" sentinel.
  boolean noneable() default false;

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
   */
  String enableOnlyWithFlag() default "";

  /**
   * If non-empty, the annotated parameter will only be present if the given semantic flag is false.
   * (If the parameter is disabled, it may not be specified by a user, and the Java method will
   * always be invoked with the parameter set to its default value.)
   *
   * <p>Note that at most one of {@link #enableOnlyWithFlag} and {@link #disableWithFlag} can be
   * non-empty.
   */
  String disableWithFlag() default "";

  /**
   * Value for the parameter when the parameter is "disabled" based on semantic flags. (When the
   * parameter is disabled, it may not be set from Starlark, but an argument of the given value is
   * passed to the annotated Java method when invoked.) (See {@link #enableOnlyWithFlag()} and
   * {@link #disableWithFlag()} for toggling a parameter with semantic flags.
   *
   * <p>The parameter value is written as a Starlark expression (for example: "False", "True", "[]",
   * "None").
   *
   * <p>This should be set (non-empty) if and only if the parameter may be disabled with a semantic
   * flag.
   *
   * <p>Note that this is very similar to {@link #defaultValue}; it may be considered "the default
   * value if no parameter is specified". It is important that this is distinct, however, in cases
   * where it is desired to have a normally-mandatory parameter toggled by flag. Such a parameter
   * should have no {@link #defaultValue} set, but should have a sensible {@link
   * #valueWhenDisabled()} set. ("unbound" may be used in cases where no value would be valid. See
   * {@link #defaultValue}.)
   */
  String valueWhenDisabled() default "";

  // TODO(bazel-team): parse the type from a single field in Starlark syntax,
  // and allow a Union as "ThisType or ThatType or NoneType":
  // String type() default "Object";
}
