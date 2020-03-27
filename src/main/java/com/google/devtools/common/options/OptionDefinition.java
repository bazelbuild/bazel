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
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Collections;
import java.util.Comparator;

/**
 * Everything the {@link OptionsParser} needs to know about how an option is defined.
 *
 * <p>An {@code OptionDefinition} is effectively a wrapper around the {@link Option} annotation and
 * the {@link Field} that is annotated, and should contain all logic about default settings and
 * behavior.
 */
public class OptionDefinition implements Comparable<OptionDefinition> {

  // TODO(b/65049598) make ConstructionException checked, which will make this checked as well.
  static class NotAnOptionException extends ConstructionException {
    NotAnOptionException(Field field) {
      super(
          "The field "
              + field.getName()
              + " does not have the right annotation to be considered an option.");
    }
  }

  /**
   * If the {@code field} is annotated with the appropriate @{@link Option} annotation, returns the
   * {@code OptionDefinition} for that option. Otherwise, throws a {@link NotAnOptionException}.
   *
   * <p>These values are cached in the {@link OptionsData} layer and should be accessed through
   * {@link OptionsParser#getOptionDefinitions(Class)}.
   */
  static OptionDefinition extractOptionDefinition(Field field) {
    Option annotation = field == null ? null : field.getAnnotation(Option.class);
    if (annotation == null) {
      throw new NotAnOptionException(field);
    }
    return new OptionDefinition(field, annotation);
  }

  private final Field field;
  private final Option optionAnnotation;
  private Converter<?> converter = null;
  private Object defaultValue = null;

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
  public Class<? extends ExpansionFunction> getExpansionFunction() {
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

  /** Returns whether an option --foo has a negative equivalent --nofoo. */
  public boolean hasNegativeOption() {
    return getType().equals(boolean.class) || getType().equals(TriState.class);
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

  /** Returns whether the arg is an expansion option. */
  public boolean hasImplicitRequirements() {
    return (getImplicitRequirements().length > 0);
  }

  /**
   * Returns whether the arg is an expansion option defined by an expansion function (and not a
   * constant expansion value).
   */
  public boolean usesExpansionFunction() {
    return getExpansionFunction() != ExpansionFunction.class;
  }

  /**
   * For an option that does not use {@link Option#allowMultiple}, returns its type. For an option
   * that does use it, asserts that the type is a {@code List<T>} and returns its element type
   * {@code T}.
   */
  Type getFieldSingularType() {
    Type fieldType = getField().getGenericType();
    if (allowsMultiple()) {
      // The validity of the converter is checked at compile time. We know the type to be
      // List<singularType>.
      ParameterizedType pfieldType = (ParameterizedType) fieldType;
      fieldType = pfieldType.getActualTypeArguments()[0];
    }
    return fieldType;
  }

  /**
   * Retrieves the {@link Converter} that will be used for this option, taking into account the
   * default converters if an explicit one is not specified.
   *
   * <p>Memoizes the converter-finding logic to avoid repeating the computation.
   */
  public Converter<?> getConverter() {
    if (converter != null) {
      return converter;
    }
    Class<? extends Converter> converterClass = getProvidedConverter();
    if (converterClass == Converter.class) {
      // No converter provided, use the default one.
      Type type = getFieldSingularType();
      converter = Converters.DEFAULT_CONVERTERS.get(type);
    } else {
      try {
        // Instantiate the given Converter class.
        Constructor<?> constructor = converterClass.getConstructor();
        converter = (Converter<?>) constructor.newInstance();
      } catch (SecurityException | IllegalArgumentException | ReflectiveOperationException e) {
        // This indicates an error in the Converter, and should be discovered the first time it is
        // used.
        throw new ConstructionException(
            String.format("Error in the provided converter for option %s", getField().getName()),
            e);
      }
    }
    return converter;
  }

  /**
   * Returns whether a field should be considered as boolean.
   *
   * <p>Can be used for usage help and controlling whether the "no" prefix is allowed.
   */
  public boolean usesBooleanValueSyntax() {
    return getType().equals(boolean.class)
        || getType().equals(TriState.class)
        || getConverter() instanceof BoolOrEnumConverter;
  }

  /** Returns the evaluated default value for this option & memoizes the result. */
  public Object getDefaultValue() {
    if (defaultValue != null) {
      return defaultValue;
    }
    // If the option allows multiple values then we intentionally return the empty list as
    // the default value of this option since it is not always the case that an option
    // that allows multiple values will have a converter that returns a list value.
    if (allowsMultiple()) {
      defaultValue = Collections.emptyList();
    } else if (isSpecialNullDefault()) {
      return null;
    } else {
      // Otherwise try to convert the default value using the converter
      Converter<?> converter = getConverter();
      String defaultValueAsString = getUnparsedDefaultValue();
      try {
        defaultValue = converter.convert(defaultValueAsString);
      } catch (OptionsParsingException e) {
        throw new ConstructionException(
            String.format(
                "OptionsParsingException while retrieving the default value for %s: %s",
                getField().getName(), e.getMessage()),
            e);
      }
    }
    return defaultValue;
  }

  /**
   * {@link OptionDefinition} is really a wrapper around a {@link Field} that caches information
   * obtained through reflection. Checking that the fields they represent are equal is sufficient
   * to check that two {@link OptionDefinition} objects are equal.
   */
  @Override
  public boolean equals(Object object) {
    if (!(object instanceof OptionDefinition)) {
      return false;
    }
    OptionDefinition otherOption = (OptionDefinition) object;
    return field.equals(otherOption.field);
  }

  @Override
  public int hashCode() {
    return field.hashCode();
  }

  @Override
  public int compareTo(OptionDefinition o) {
    return getOptionName().compareTo(o.getOptionName());
  }

  @Override
  public String toString() {
    return String.format("option '--%s'", getOptionName());
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
