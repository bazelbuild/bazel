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
package com.google.devtools.common.options;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * An interface for annotating fields in classes (derived from OptionsBase)
 * that are options.
 */
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Option {

  /**
   * The name of the option ("--name").
   */
  String name();

  /**
   * The single-character abbreviation of the option ("-abbrev").
   */
  char abbrev() default '\0';

  /**
   * A help string for the usage information.
   */
  String help() default "";

  /**
   * The default value for the option. This method should only be invoked
   * directly by the parser implementation. Any access to default values
   * should go via the parser to allow for application specific defaults.
   *
   * <p>There are two reasons this is a string.  Firstly, it ensures that
   * explicitly specifying this option at its default value (as printed in the
   * usage message) has the same behavior as not specifying the option at all;
   * this would be very hard to achieve if the default value was an instance of
   * type T, since we'd need to ensure that {@link #toString()} and {@link
   * #converter} were dual to each other.  The second reason is more mundane
   * but also more restrictive: annotation values must be compile-time
   * constants.
   *
   * <p>If an option's defaultValue() is the string "null", the option's
   * converter will not be invoked to interpret it; a null reference will be
   * used instead.  (It would be nice if defaultValue could simply return null,
   * but bizarrely, the Java Language Specification does not consider null to
   * be a compile-time constant.)  This special interpretation of the string
   * "null" is only applicable when computing the default value; if specified
   * on the command-line, this string will have its usual literal meaning.
   *
   * <p>The default value for flags that set allowMultiple is always the empty
   * list and its default value is ignored.
   */
  String defaultValue();

  /**
   * A string describing the category of options that this belongs to. {@link
   * OptionsParser#describeOptions} prints options of the same category grouped
   * together.
   */
  String category() default "misc";

  /**
   * The converter that we'll use to convert this option into an object or
   * a simple type. The default is to use the builtin converters.
   * Custom converters must implement the {@link Converter} interface.
   */
  @SuppressWarnings({"unchecked", "rawtypes"})
  // Can't figure out how to coerce Converter.class into Class<? extends Converter<?>>
  Class<? extends Converter> converter() default Converter.class;

  /**
   * A flag indicating whether the option type should be allowed to occur
   * multiple times in a single option list.
   *
   * <p>If the command can occur multiple times, then the attribute value
   * <em>must</em> be a list type {@code List<T>}, and the result type of the
   * converter for this option must either match the parameter {@code T} or
   * {@code List<T>}. In the latter case the individual lists are concatenated
   * to form the full options value.
   *
   * <p>The {@link #defaultValue()} field of the annotation is ignored for repeatable
   * flags and the default value will be the empty list.
   */
  boolean allowMultiple() default false;

  /**
   * If the option is actually an abbreviation for other options, this field will
   * contain the strings to expand this option into. The original option is dropped
   * and the replacement used in its stead. It is recommended that such an option be
   * of type {@link Void}.
   *
   * An expanded option overrides previously specified options of the same name,
   * even if it is explicitly specified. This is the original behavior and can
   * be surprising if the user is not aware of it, which has led to several
   * requests to change this behavior. This was discussed in the blaze team and
   * it was decided that it is not a strong enough case to change the behavior.
   */
  String[] expansion() default {};

  /**
   * If the option requires that additional options be implicitly appended, this field
   * will contain the additional options. Implicit dependencies are parsed at the end
   * of each {@link OptionsParser#parse} invocation, and override options specified in
   * the same call. However, they can be overridden by options specified in a later
   * call or by options with a higher priority.
   *
   * @see OptionPriority
   */
  String[] implicitRequirements() default {};

  /**
   * If this field is a non-empty string, the option is deprecated, and a
   * deprecation warning is added to the list of warnings when such an option
   * is used.
   */
  String deprecationWarning() default "";

  /**
   * The old name for this option. If an option has a name "foo" and an old name "bar",
   * --foo=baz and --bar=baz will be equivalent. If the old name is used, a warning will be printed
   * indicating that the old name is deprecated and the new name should be used.
   */
  String oldName() default "";

  /**
   * Indicates that this option is a wrapper for other options, and will be unwrapped
   * when parsed. For example, if foo is a wrapper option, then "--foo=--bar=baz"
   * will be parsed as the flag "--bar=baz" (rather than --foo taking the value
   * "--bar=baz"). A wrapper option should have the type {@link Void} (if it is something other
   * than Void, the parser will not assign a value to it). The
   * {@link Option#implicitRequirements()}, {@link Option#expansion()}, {@link Option#converter()}
   * attributes will not be processed. Wrapper options are implicitly repeatable (i.e., as though
   * {@link Option#allowMultiple()} is true regardless of its value in the annotation).
   */
  boolean wrapperOption() default false;
}
