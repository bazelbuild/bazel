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

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.List;
import javax.annotation.Nullable;

/**
 * A {@code FieldOptionDefinition} is effectively a wrapper around the {@link Option} annotation and
 * the {@link Field} that is annotated, and should contain all logic about default settings and
 * behavior.
 */
public class FieldOptionDefinition extends OptionDefinition {

  /**
   * A special value used to specify an absence of default value.
   *
   * @see Option#defaultValue
   */
  public static final String SPECIAL_NULL_DEFAULT_VALUE = "null";

  /** Exception used when trying to create a {@link FieldOptionDefinition} for an invalid field. */
  // TODO(b/65049598) make ConstructionException checked, which will make this checked as well.
  public static class NotAnOptionException extends ConstructionException {
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
  static FieldOptionDefinition extractOptionDefinition(Field field) {
    Option annotation = field == null ? null : field.getAnnotation(Option.class);
    if (annotation == null) {
      throw new NotAnOptionException(field);
    }
    return new FieldOptionDefinition(field, annotation);
  }

  /**
   * Wraps a converted default value into a {@link List} if the converter doesn't do it on its own.
   *
   * <p>This is to make sure multiple ({@link Option#allowMultiple()}) options' default values are
   * always converted to a list representation.
   *
   * <p>In general it mimics the {@link RepeatableOptionValueDescription#addOptionInstance}
   * behavior: multiple option default value is treated as if it appeared on the command line only
   * once with the specified value.
   *
   * <p>Note that on a command line multiple options can appear multiple times while each can
   * support multiple values (e.g. comma-separated - depending on a converter). Thus default value
   * for multiple option is (depending on the converter) a strict subset of the set of potential
   * values for the option.
   */
  @SuppressWarnings("unchecked") // Not an unchecked cast - there's an explicit type check before it
  static List<Object> maybeWrapMultipleDefaultValue(Object convertedDefaultValue) {
    if (convertedDefaultValue instanceof List) {
      return (List<Object>) convertedDefaultValue;
    } else {
      return Arrays.asList(convertedDefaultValue);
    }
  }

  private final Field field;
  private final Option optionAnnotation;
  private volatile Converter<?> converter = null;
  private volatile Object defaultValue = null;

  private FieldOptionDefinition(Field field, Option optionAnnotation) {
    this.field = field;
    this.optionAnnotation = optionAnnotation;
  }

  /** Returns the underlying {@code field} for this {@code OptionDefinition}. */
  protected Field getField() {
    return field;
  }

  @Override
  public <C extends OptionsBase> Class<? extends C> getDeclaringClass(Class<C> baseClass) {
    Class<?> declaringClass = field.getDeclaringClass();
    if (!baseClass.isAssignableFrom(declaringClass)) {
      throw new IllegalStateException(
          String.format(
              "Declaring class %s is not assignable from requested base class %s",
              declaringClass, baseClass));
    }
    @SuppressWarnings("unchecked") // This should be safe based on the previous check.
    Class<? extends C> castClass = (Class<? extends C>) declaringClass;
    return castClass;
  }

  @Override
  public Object getRawValue(OptionsBase optionsBase) {
    try {
      return field.get(optionsBase);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException(
          String.format(
              "Unexpected illegal access trying to fetch value for field %s in options %s: ",
              this.getOptionName(), optionsBase),
          e);
    }
  }

  @Override
  public Object getValue(OptionsBase optionsBase) {
    Object value = getRawValue(optionsBase);
    if (value == null && !isSpecialNullDefault()) {
      // See {@link Option#defaultValue} for an explanation of default "null" strings.
      value = getUnparsedDefaultValue();
    }
    return value;
  }

  @Override
  public void setValue(OptionsBase optionsBase, Object value) {
    try {
      field.set(optionsBase, value);
    } catch (IllegalAccessException e) {
      throw new IllegalStateException("Couldn't set " + this.getOptionName(), e);
    }
  }

  @Override
  public boolean isDeprecated() {
    return field.isAnnotationPresent(Deprecated.class);
  }

  /**
   * Returns the name of the option ("--name").
   *
   * <p>Labelled "Option" name to distinguish it from the field's name.
   */
  @Override
  public String getOptionName() {
    return optionAnnotation.name();
  }

  /** The single-character abbreviation of the option ("-a"). */
  @Override
  public char getAbbreviation() {
    return optionAnnotation.abbrev();
  }

  /** {@link Option#help()} */
  @Override
  public String getHelpText() {
    return optionAnnotation.help();
  }

  /** {@link Option#valueHelp()} */
  @Override
  public String getValueTypeHelpText() {
    return optionAnnotation.valueHelp();
  }

  /** {@link Option#defaultValue()} */
  @Override
  public String getUnparsedDefaultValue() {
    return optionAnnotation.defaultValue();
  }

  /** {@link Option#category()} */
  @Override
  public String getOptionCategory() {
    return optionAnnotation.category();
  }

  /** {@link Option#documentationCategory()} */
  @Override
  public OptionDocumentationCategory getDocumentationCategory() {
    return optionAnnotation.documentationCategory();
  }

  /** {@link Option#effectTags()} */
  @Override
  public OptionEffectTag[] getOptionEffectTags() {
    return optionAnnotation.effectTags();
  }

  /** {@link Option#metadataTags()} */
  @Override
  public OptionMetadataTag[] getOptionMetadataTags() {
    return optionAnnotation.metadataTags();
  }

  /** {@link Option#converter()} ()} */
  @Override
  @SuppressWarnings({"rawtypes"})
  public Class<? extends Converter> getProvidedConverter() {
    return optionAnnotation.converter();
  }

  /** {@link Option#allowMultiple()} */
  @Override
  public boolean allowsMultiple() {
    return optionAnnotation.allowMultiple();
  }

  /** {@link Option#expansion()} */
  @Override
  public String[] getOptionExpansion() {
    return optionAnnotation.expansion();
  }

  /** {@link Option#implicitRequirements()} ()} */
  @Override
  public String[] getImplicitRequirements() {
    return optionAnnotation.implicitRequirements();
  }

  /** {@link Option#deprecationWarning()} ()} */
  @Override
  public String getDeprecationWarning() {
    return optionAnnotation.deprecationWarning();
  }

  /** {@link Option#oldName()} ()} ()} */
  @Override
  public String getOldOptionName() {
    return optionAnnotation.oldName();
  }

  /** {@link Option#oldNameWarning()} */
  @Override
  public boolean getOldNameWarning() {
    return optionAnnotation.oldNameWarning();
  }

  @Override
  public Class<?> getType() {
    return field.getType();
  }

  @Override
  public boolean isSpecialNullDefault() {
    return SPECIAL_NULL_DEFAULT_VALUE.equals(getUnparsedDefaultValue()) && !getType().isPrimitive();
  }

  /**
   * For an option that does not use {@link Option#allowMultiple}, returns its type. For an option
   * that does use it, asserts that the type is a {@code List<T>} and returns its element type
   * {@code T}.
   */
  @Override
  public Type getFieldSingularType() {
    Type fieldType = field.getGenericType();
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
  @Override
  public Converter<?> getConverter() {
    if (converter != null) {
      return converter;
    }

    synchronized (this) {
      if (converter != null) {
        return converter;
      }

      @SuppressWarnings("rawtypes")
      Class<? extends Converter> converterClass = getProvidedConverter();
      if (converterClass == Converter.class) {
        // No converter provided, use the default one.
        Type type = getFieldSingularType();
        converter = Converters.DEFAULT_CONVERTERS.get(type);
      } else {
        try {
          // Instantiate the given Converter class.
          Constructor<?> constructor = converterClass.getDeclaredConstructor();
          constructor.setAccessible(true);
          converter = (Converter<?>) constructor.newInstance();
        } catch (SecurityException | IllegalArgumentException | ReflectiveOperationException e) {
          // This indicates an error in the Converter, and should be discovered the first time it is
          // used.
          throw new ConstructionException(
              String.format("Error in the provided converter for option %s", field.getName()), e);
        }
      }
      return converter;
    }
  }

  /** Returns the evaluated default value for this option & memoizes the result. */
  @Override
  @Nullable
  public Object getDefaultValue(@Nullable Object conversionContext) {
    if (defaultValue != null) {
      return defaultValue;
    }

    synchronized (this) {
      if (defaultValue != null) {
        return defaultValue;
      }

      if (isSpecialNullDefault()) {
        return allowsMultiple() ? ImmutableList.of() : null;
      }

      Converter<?> converter = getConverter();
      String defaultValueAsString = getUnparsedDefaultValue();
      try {
        Object convertedDefaultValue = converter.convert(defaultValueAsString, conversionContext);
        defaultValue =
            allowsMultiple()
                ? maybeWrapMultipleDefaultValue(convertedDefaultValue)
                : convertedDefaultValue;
      } catch (OptionsParsingException e) {
        throw new ConstructionException(
            String.format(
                "OptionsParsingException while retrieving the default value for %s: %s",
                field.getName(), e.getMessage()),
            e);
      }

      return defaultValue;
    }
  }

  /**
   * {@link FieldOptionDefinition} is really a wrapper around a {@link Field} that caches
   * information obtained through reflection. Checking that the fields they represent are equal is
   * sufficient to check that two {@link FieldOptionDefinition} objects are equal.
   */
  @Override
  public boolean equals(Object object) {
    if (!(object instanceof FieldOptionDefinition)) {
      return false;
    }
    FieldOptionDefinition otherOption = (FieldOptionDefinition) object;
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
}
