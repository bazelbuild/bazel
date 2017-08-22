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

package com.google.devtools.common.options;

import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Field;
import java.util.Comparator;

/**
 * Everything the {@link OptionsParser} needs to know about how an option is defined.
 *
 * <p>An {@code OptionDefinition} is effectively a wrapper around the {@link Option} annotation and
 * the {@link Field} that is annotated, and should contain all logic about default settings and
 * behavior.
 */
public final class OptionDefinition {

  /**
   * If the {@code field} is annotated with the appropriate @{@link Option} annotation, returns the
   * {@code OptionDefinition} for that option. Otherwise, throws a {@link ConstructionException}.
   */
  public static OptionDefinition extractOptionDefinition(Field field) {
    Option annotation = field == null ? null : field.getAnnotation(Option.class);
    if (annotation == null) {
      throw new ConstructionException(
          "The field " + field + " does not have the right annotation to be considered an option.");
    }
    return new OptionDefinition(field, annotation);
  }

  private final Field field;
  private final Option optionAnnotation;

  private OptionDefinition(Field field, Option optionAnnotation) {
    this.field = field;
    this.optionAnnotation = optionAnnotation;
  }

  /** Returns the underlying {@code field} for this {@code OptionDefinition}. */
  public Field getField() {
    return field;
  }

  /**
   * Returns the name of the option ("--name").
   *
   * <p>Labelled "Option" name to distinguish it from the field's name.
   */
  public String getOptionName() {
    return optionAnnotation.name();
  }

  /** The single-character abbreviation of the option ("-a"). */
  public char getAbbreviation() {
    return optionAnnotation.abbrev();
  }

  /** {@link Option#help()} */
  public String getHelpText() {
    return optionAnnotation.help();
  }

  /** {@link Option#valueHelp()} */
  public String getValueTypeHelpText() {
    return optionAnnotation.valueHelp();
  }

  /** {@link Option#defaultValue()} */
  public String getUnparsedDefaultValue() {
    return optionAnnotation.defaultValue();
  }

  /** {@link Option#category()} */
  public String getOptionCategory() {
    return optionAnnotation.category();
  }

  /** {@link Option#documentationCategory()} */
  public OptionDocumentationCategory getDocumentationCategory() {
    return optionAnnotation.documentationCategory();
  }

  /** {@link Option#effectTags()} */
  public OptionEffectTag[] getOptionEffectTags() {
    return optionAnnotation.effectTags();
  }

  /** {@link Option#metadataTags()} */
  public OptionMetadataTag[] getOptionMetadataTags() {
    return optionAnnotation.metadataTags();
  }

  /** {@link Option#converter()} ()} */
  @SuppressWarnings({"rawtypes"})
  public Class<? extends Converter> getProvidedConverter() {
    return optionAnnotation.converter();
  }

  /** {@link Option#allowMultiple()} */
  public boolean allowsMultiple() {
    return optionAnnotation.allowMultiple();
  }

  /** {@link Option#expansion()} */
  public String[] getOptionExpansion() {
    return optionAnnotation.expansion();
  }

  /** {@link Option#expansionFunction()} ()} */
  Class<? extends ExpansionFunction> getExpansionFunction() {
    return optionAnnotation.expansionFunction();
  }

  /** {@link Option#implicitRequirements()} ()} */
  public String[] getImplicitRequirements() {
    return optionAnnotation.implicitRequirements();
  }

  /** {@link Option#deprecationWarning()} ()} */
  public String getDeprecationWarning() {
    return optionAnnotation.deprecationWarning();
  }

  /** {@link Option#oldName()} ()} ()} */
  public String getOldOptionName() {
    return optionAnnotation.oldName();
  }

  /** {@link Option#wrapperOption()} ()} ()} */
  public boolean isWrapperOption() {
    return optionAnnotation.wrapperOption();
  }

  /** The type of the optionDefinition. */
  public Class<?> getType() {
    return field.getType();
  }

  /** Whether this field has type Void. */
  boolean isVoidField() {
    return getType().equals(Void.class);
  }

  public boolean isSpecialNullDefault() {
    return getUnparsedDefaultValue().equals("null") && !getType().isPrimitive();
  }

  /** Returns whether the arg is an expansion option. */
  public boolean isExpansionOption() {
    return (getOptionExpansion().length > 0 || usesExpansionFunction());
  }

  /**
   * Returns whether the arg is an expansion option defined by an expansion function (and not a
   * constant expansion value).
   */
  public boolean usesExpansionFunction() {
    return getExpansionFunction() != ExpansionFunction.class;
  }

  static final Comparator<OptionDefinition> BY_OPTION_NAME =
      Comparator.comparing(OptionDefinition::getOptionName);

  /**
   * An ordering relation for option-field fields that first groups together options of the same
   * category, then sorts by name within the category.
   */
  static final Comparator<OptionDefinition> BY_CATEGORY =
      (left, right) -> {
        int r = left.getOptionCategory().compareTo(right.getOptionCategory());
        return r == 0 ? BY_OPTION_NAME.compare(left, right) : r;
      };
}
