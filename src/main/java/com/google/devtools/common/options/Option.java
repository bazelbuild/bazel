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
 * An interface for annotating fields in classes (derived from OptionsBase) that are options.
 *
 * <p>The fields of this annotation have matching getters in {@link OptionDefinition}. Please do not
 * access these fields directly, but instead go through that class.
 *
 * <p>A number of checks are run on an Option's fields' values at compile time. See {@link
 * com.google.devtools.common.options.processor.OptionProcessor} for details.
 */
@Target(ElementType.FIELD)
@Retention(RetentionPolicy.RUNTIME)
public @interface Option {
  /** The name of the option ("--name"). */
  String name();

  /** The single-character abbreviation of the option ("-a"). */
  char abbrev() default '\0';

  /**
   * A help string for the usage information. Note that this should be in plain text (no HTML tags,
   * for example).
   */
  String help() default "";

  /**
   * A short text string to describe the type of the expected value. E.g., <code>regex</code>. This
   * is ignored for boolean, tristate, boolean_or_enum, and void options.
   */
  String valueHelp() default "";

  /**
   * The default value for the option. This method should only be invoked directly by the parser
   * implementation. Any access to default values should go via the parser to allow for application
   * specific defaults.
   *
   * <p>There are two reasons this is a string. Firstly, it ensures that explicitly specifying this
   * option at its default value (as printed in the usage message) has the same behavior as not
   * specifying the option at all; this would be very hard to achieve if the default value was an
   * instance of type T, since we'd need to ensure that {@link #toString()} and {@link #converter}
   * were dual to each other. The second reason is more mundane but also more restrictive:
   * annotation values must be compile-time constants.
   *
   * <p>If an option's defaultValue() is the string "null" (see {@link
   * OptionDefinition#SPECIAL_NULL_DEFAULT_VALUE}), the option's converter will not be invoked to
   * interpret it; an empty {@link java.util.List} (for {@code allowMultiple = true} options) or a
   * null reference (for others) will be used instead. (It would be nice if defaultValue could
   * simply return null, but bizarrely, the Java Language Specification does not consider null to be
   * a compile-time constant.) This special interpretation of the string "null" is only applicable
   * when computing the default value; if specified on the command-line, this string will have its
   * usual literal meaning.
   *
   * <p>Multiple options (e.g. with {@code allowMultiple = true}) are not allowed to have default
   * values (with only a small number of exceptions - see {@link OptionsProcessor}), thus should
   * always use {@link OptionDefinition#SPECIAL_NULL_DEFAULT_VALUE}.
   */
  String defaultValue();

  /**
   * This category field is deprecated. Bazel is in the process of migrating all options to use the
   * better defined enums in OptionDocumentationCategory and the tags in the option_filters.proto
   * file. It will still be used for the usage documentation until a sufficient proportion of
   * options are using the new system.
   *
   * <p>Please leave the old category field in existing options to minimize disruption to the Help
   * output during the transition period. All uses of this field will be removed when transition is
   * complete. This category field has no effect on the other fields below, having both set is not a
   * problem.
   */
  @Deprecated
  String category() default "misc";

  /**
   * Grouping categories used for usage documentation. See the enum's definition for details.
   *
   * <p>For undocumented flags that aren't listed anywhere, set this to
   * OptionDocumentationCategory.UNDOCUMENTED.
   */
  OptionDocumentationCategory documentationCategory();

  /**
   * Tag about the intent or effect of this option. Unless this option is a no-op (and the reason
   * for this should be documented) all options should have some effect, so this needs to have at
   * least one value, and as many as apply.
   *
   * <p>No option should list NO_OP or UNKNOWN with other effects listed, but all other combinations
   * are allowed.
   */
  OptionEffectTag[] effectTags();

  /**
   * Tag about the option itself, not its effect, such as option state (experimental) or intended
   * use (a value that isn't a flag but is used internally, for example, is "internal")
   *
   * <p>If one or more of the OptionMetadataTag values apply, please include, but otherwise, this
   * list can be left blank.
   *
   * <p>Hidden or internal options must be UNDOCUMENTED (set in {@link #documentationCategory()}).
   */
  OptionMetadataTag[] metadataTags() default {};

  /**
   * The converter that we'll use to convert the string representation of this option's value into
   * an object or a simple type. The default is to use the builtin converters ({@link
   * Converters#DEFAULT_CONVERTERS}). Custom converters must implement the {@link Converter}
   * interface.
   */
  @SuppressWarnings({"unchecked", "rawtypes"})
  // Can't figure out how to coerce Converter.class into Class<? extends Converter<?>>
  Class<? extends Converter> converter() default Converter.class;

  /**
   * A boolean value indicating whether the option type should be allowed to occur multiple times in
   * a single arg list.
   *
   * <p>If the option can occur multiple times, then the attribute value <em>must</em> be a list
   * type {@code List<T>}, and the result type of the converter for this option must either match
   * the parameter {@code T} or {@code List<T>}. In the latter case the individual lists are
   * concatenated to form the full options value.
   *
   * <p>The {@link #defaultValue()} field of the annotation is ignored for repeatable flags and the
   * default value will be the empty list.
   */
  boolean allowMultiple() default false;

  /**
   * If the option is actually an abbreviation for other options, this field will contain the
   * strings to expand this option into. The original option is dropped and the replacement used in
   * its stead. It is recommended that such an option be of type {@link Void}.
   *
   * <p>An expanded option overrides previously specified options of the same name, even if it is
   * explicitly specified. This is the original behavior and can be surprising if the user is not
   * aware of it, which has led to several requests to change this behavior. This was discussed in
   * the blaze team and it was decided that it is not a strong enough case to change the behavior.
   */
  String[] expansion() default {};

  /**
   * A mechanism for specifying an expansion that is a function of the parser's {@link
   * IsolatedOptionsData}. This can be used to create an option that expands to different strings
   * depending on what other options the parser knows about.
   *
   * <p>If provided (i.e. not {@link ExpansionFunction}{@code .class}), the {@code expansion} field
   * must not be set. The mechanism of expansion is as if the {@code expansion} field were set to
   * whatever the return value of this function is.
   */
  Class<? extends ExpansionFunction> expansionFunction() default ExpansionFunction.class;

  /**
   * Additional options that need to be implicitly added for this option.
   *
   * <p>Nothing guarantees that these options are not overridden by later or higher-priority values
   * for the same options, so if this is truly a requirement, the user should check that the correct
   * set of options is set.
   *
   * <p>These requirements are added for ANY mention of this option, so may not work as intended: in
   * the case where a user is trying to explicitly turn off a flag (say, by setting a boolean flag
   * to its default value of false), the mention will still turn on its requirements. For this
   * reason, it is best not to use this feature, and rely on expansion flags if multi-flag groupings
   * are needed.
   */
  String[] implicitRequirements() default {};

  /**
   * If this field is a non-empty string, the option is deprecated, and a deprecation warning is
   * added to the list of warnings when such an option is used.
   */
  String deprecationWarning() default "";

  /**
   * The old name for this option. If an option has a name "foo" and an old name "bar", --foo=baz
   * and --bar=baz will be equivalent. If the old name is used, a warning will be printed indicating
   * that the old name is deprecated and the new name should be used.
   */
  String oldName() default "";
}
