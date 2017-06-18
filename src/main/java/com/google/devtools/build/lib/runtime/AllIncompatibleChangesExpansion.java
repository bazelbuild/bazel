// Copyright 2017 The Bazel Authors. All rights reserved.
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

package com.google.devtools.build.lib.runtime;

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.Converter;
import com.google.devtools.common.options.ExpansionFunction;
import com.google.devtools.common.options.IsolatedOptionsData;
import com.google.devtools.common.options.Option;
import com.google.devtools.common.options.OptionsBase;
import com.google.devtools.common.options.OptionsParser;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Map;

/**
 * Expansion function for {@code --all_incompatible_changes}. Expands to all options of form {@code
 * --incompatible_*} that are declared in the {@link OptionsBase} subclasses that are passed to the
 * parser.
 *
 * <p>The incompatible changes system provides users with a uniform way of opting into backwards-
 * incompatible changes, in order to test whether their builds will be broken by an upcoming
 * release. When adding a new breaking change to Bazel, prefer to use this mechanism for guarding
 * the behavior.
 *
 * <p>An {@link Option}-annotated field that is considered an incompatible change must satisfy the
 * following requirements.
 *
 * <ul>
 *   <li>the {@link Option#name} must be prefixed with "incompatible_"
 *   <li>the {@link Option#category} must be "incompatible changes"
 *   <li>the {@link Option#help} field must be set, and must refer the user to information about
 *       what the change does and how to migrate their code
 *   <li>the following fields may not be used: {@link Option#abbrev}, {@link Option#valueHelp},
 *       {@link Option#converter}, {@link Option#allowMultiple}, {@link Option#oldName}, and {@link
 *       Option#wrapperOption}
 * </ul>
 *
 * Example:
 *
 * <pre>{@code
 * @Option(
 *   name = "incompatible_foo",
 *   category = "incompatible changes",
 *   defaultValue = "false",
 *   help = "Deprecates bar and changes the semantics of baz. To migrate your code see [...].")
 * public boolean incompatibleFoo;
 * }</pre>
 *
 * All options that satisfy either the name or category requirement will be validated using the
 * above criteria. Any failure will cause {@link IllegalArgumentException} to be thrown, which will
 * cause the construction of the {@link OptionsParser} to fail with the <i>unchecked</i> exception
 * {@link OptionsParser.ConstructionException}. Therefore, when adding a new incompatible change, be
 * aware that an error in the specification of the {@code @Option} will exercise failure code paths
 * in the early part of the Bazel server execution.
 *
 * <p>After the breaking change has been enabled by default, it is recommended (required?) that the
 * flag stick around for a few releases, to provide users the flexibility to opt out. Even after
 * enabling the behavior unconditionally, it can still be useful to keep the flag around as a valid
 * no-op so that Bazel invocations are not immediately broken.
 *
 * <p>Generally speaking, we should never reuse names for multiple options. Therefore, when choosing
 * a name for a new incompatible change, try to describe not just the affected feature, but what the
 * change to that feature is. This avoids conflicts in case the feature changes multiple times. For
 * example, {@code "--incompatible_depset_constructor"} is ambiguous because it only communicates
 * that there is a change to how depsets are constructed, but {@code
 * "--incompatible_disallow_set_constructor"} uniquely says that the {@code set} alias for the
 * depset constructor is being disallowed.
 */
// Javadoc can't resolve inner classes.
@SuppressWarnings("javadoc")
public class AllIncompatibleChangesExpansion implements ExpansionFunction {

  // The reserved prefix for all incompatible change option names.
  public static final String INCOMPATIBLE_NAME_PREFIX = "incompatible_";
  // The reserved category for all incompatible change options.
  public static final String INCOMPATIBLE_CATEGORY = "incompatible changes";

  /**
   * Ensures that the given option satisfies all the requirements on incompatible change options
   * enumerated above.
   *
   * <p>If any of these requirements are not satisfied, {@link IllegalArgumentException} is thrown,
   * as this constitutes an internal error in the declaration of the option.
   */
  private static void validateIncompatibleChange(Field field, Option annotation) {
    String prefix = "Incompatible change option '--" + annotation.name() + "' ";

    // To avoid ambiguity, and the suggestion of using .isEmpty().
    String defaultString = "";

    // Validate that disallowed fields aren't used. These will need updating if the default values
    // in Option ever change, and perhaps if new fields are added.
    if (annotation.abbrev() != '\0') {
      throw new IllegalArgumentException(prefix + "must not use the abbrev field");
    }
    if (!annotation.valueHelp().equals(defaultString)) {
      throw new IllegalArgumentException(prefix + "must not use the valueHelp field");
    }
    if (annotation.converter() != Converter.class) {
      throw new IllegalArgumentException(prefix + "must not use the converter field");
    }
    if (annotation.allowMultiple()) {
      throw new IllegalArgumentException(prefix + "must not use the allowMultiple field");
    }
    if (annotation.implicitRequirements().length > 0) {
      throw new IllegalArgumentException(prefix + "must not use the implicitRequirements field");
    }
    if (!annotation.oldName().equals(defaultString)) {
      throw new IllegalArgumentException(prefix + "must not use the oldName field");
    }
    if (annotation.wrapperOption()) {
      throw new IllegalArgumentException(prefix + "must not use the wrapperOption field");
    }

    // Validate the fields that are actually allowed.
    if (!annotation.name().startsWith(INCOMPATIBLE_NAME_PREFIX)) {
      throw new IllegalArgumentException(prefix + "must have name starting with \"incompatible_\"");
    }
    if (!annotation.category().equals(INCOMPATIBLE_CATEGORY)) {
      throw new IllegalArgumentException(prefix + "must have category \"incompatible changes\"");
    }
    if (!IsolatedOptionsData.isExpansionOption(annotation)) {
      if (!field.getType().equals(Boolean.TYPE)) {
        throw new IllegalArgumentException(
            prefix + "must have boolean type (unless it's an expansion option)");
      }
    }
    if (annotation.help().equals(defaultString)) {
      throw new IllegalArgumentException(
          prefix
              + "must have a \"help\" string that refers the user to "
              + "information about this change and how to migrate their code");
    }
  }

  @Override
  public ImmutableList<String> getExpansion(IsolatedOptionsData optionsData) {
    // Grab all registered options that are identified as incompatible changes by either name or
    // by category. Ensure they satisfy our requirements.
    ArrayList<String> incompatibleChanges = new ArrayList<>();
    for (Map.Entry<String, Field> entry : optionsData.getAllNamedFields()) {
      Field field = entry.getValue();
      Option annotation = field.getAnnotation(Option.class);
      if (annotation.name().startsWith(INCOMPATIBLE_NAME_PREFIX)
          || annotation.category().equals(INCOMPATIBLE_CATEGORY)) {
        validateIncompatibleChange(field, annotation);
        incompatibleChanges.add("--" + annotation.name());
      }
    }
    // Sort to get a deterministic canonical order. This probably isn't necessary because the
    // options parser will do its own sorting when canonicalizing, but it seems like it can't hurt.
    incompatibleChanges.sort(null);
    return ImmutableList.copyOf(incompatibleChanges);
  }
}
