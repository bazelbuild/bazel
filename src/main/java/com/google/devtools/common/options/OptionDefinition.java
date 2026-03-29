// Copyright 2024 The Bazel Authors. All rights reserved.
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

import static java.util.Comparator.comparing;

import com.google.common.collect.ImmutableList;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Member;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import javax.annotation.Nullable;

/** Everything the {@link OptionsParser} needs to know about how an option is defined. */
public abstract class OptionDefinition implements Comparable<OptionDefinition> {

  /**
   * A special value used to specify an absence of default value.
   *
   * @see Option#defaultValue
   */
  public static final String SPECIAL_NULL_DEFAULT_VALUE = "null";

  /** Exception used when trying to create an {@link OptionDefinition} for an invalid member. */
  // TODO(b/65049598) make ConstructionException checked, which will make this checked as well.
  public static class NotAnOptionException extends ConstructionException {
    public NotAnOptionException(Member member) {
      super(
          String.format(
              "The %s %s does not have the right annotation to be considered an option.",
              member instanceof Field ? "field" : "method", member.getName()));
    }
  }

  /** An ordering relation for options that orders by the option name. */
  public static final Comparator<OptionDefinition> BY_OPTION_NAME =
      Comparator.comparing(OptionDefinition::getOptionName);

  /**
   * An ordering relation for options that first groups together options of the same category, then
   * sorts by name within the category.
   */
  public static final Comparator<OptionDefinition> BY_CATEGORY =
      comparing(OptionDefinition::getOptionCategory).thenComparing(BY_OPTION_NAME);

  protected final Option optionAnnotation;
  private volatile Converter<?> converter = null;
  private volatile Object defaultValue = null;

  protected OptionDefinition(Option optionAnnotation) {
    this.optionAnnotation = optionAnnotation;
  }

  /** Returns the declaring {@link OptionsBase} class that owns this option. */
  public abstract <C extends OptionsBase> Class<? extends C> getDeclaringClass(Class<C> baseClass);

  /**
   * Returns the raw value of the option. Use {@link #getValue} if possible to correctly handle
   * default values.
   */
  public abstract Object getRawValue(OptionsBase optionsBase);

  /** Returns the value of this option, taking default values into account. */
  public Object getValue(OptionsBase optionsBase) {
    Object value = getRawValue(optionsBase);
    if (value == null && !isSpecialNullDefault()) {
      value = getUnparsedDefaultValue();
    }
    return value;
  }

  /**
   * Returns the value of this option as a boolean. If the option is not boolean-typed, throws an
   * IllegalStateException.
   */
  public boolean getBooleanValue(OptionsBase optionsBase) {
    // Check for primitive boolean first, as it's more common.
    if (!getType().isAssignableFrom(Boolean.TYPE) && !getType().isAssignableFrom(Boolean.class)) {
      throw new IllegalStateException(
          "Option "
              + getOptionName()
              + " is not a boolean, has type "
              + getType().getCanonicalName());
    }
    return getValue(optionsBase).equals(Boolean.TRUE);
  }

  /** Sets the value for this option. */
  public abstract void setValue(OptionsBase optionsBase, Object value);

  /** Returns whether this option is deprecated. */
  public abstract boolean isDeprecated();

  /** Returns the name of this option. */
  public String getOptionName() {
    return optionAnnotation.name();
  }

  /** Returns a one-character abbreviation for this option, if any. */
  public char getAbbreviation() {
    return optionAnnotation.abbrev();
  }

  /** Returns the help test for this option. */
  public String getHelpText() {
    return optionAnnotation.help();
  }

  /** Returns a short description of the expected type of this option. */
  public String getValueTypeHelpText() {
    return optionAnnotation.valueHelp();
  }

  /**
   * Returns the default value of this option, with no conversion performed. Should only be used by
   * the parser.
   */
  public String getUnparsedDefaultValue() {
    return optionAnnotation.defaultValue();
  }

  /**
   * Returns the deprecated option category.
   *
   * @deprecated Use {@link #getDocumentationCategory} instead
   */
  @Deprecated
  public String getOptionCategory() {
    return optionAnnotation.category();
  }

  /** Returns the option category. */
  public OptionDocumentationCategory getDocumentationCategory() {
    return optionAnnotation.documentationCategory();
  }

  /** Returns data about the intended effects of this option. */
  public OptionEffectTag[] getOptionEffectTags() {
    return optionAnnotation.effectTags();
  }

  /** Returns metadata about this option. */
  public OptionMetadataTag[] getOptionMetadataTags() {
    return optionAnnotation.metadataTags();
  }

  /** Returns a converter to use for this option. */
  @SuppressWarnings({"rawtypes"})
  public Class<? extends Converter> getProvidedConverter() {
    return optionAnnotation.converter();
  }

  /** Returns whether this option allows multiple instances to be combined into a list. */
  public boolean allowsMultiple() {
    return optionAnnotation.allowMultiple();
  }

  /** Returns any options which are added if this option is present. */
  public String[] getOptionExpansion() {
    return optionAnnotation.expansion();
  }

  /** Returns additional options that need to be implicitly added for this option. */
  public String[] getImplicitRequirements() {
    return optionAnnotation.implicitRequirements();
  }

  /** Returns a deprecation warning for this option, if one is present. */
  public String getDeprecationWarning() {
    return optionAnnotation.deprecationWarning();
  }

  /** Returns the old name for this option, if one is present. */
  public String getOldOptionName() {
    return optionAnnotation.oldName();
  }

  /** Returns a warning to use with this option if the old name is specified. */
  public boolean getOldNameWarning() {
    return optionAnnotation.oldNameWarning();
  }

  /** The type of the optionDefinition. */
  public abstract Class<?> getType();

  /** Whether this field has type Void. */
  public boolean isVoidField() {
    return getType().equals(Void.class);
  }

  // TODO: blaze-configurability - try to remove special handling for defaults
  public boolean isSpecialNullDefault() {
    return getUnparsedDefaultValue().equals(SPECIAL_NULL_DEFAULT_VALUE) && !getType().isPrimitive();
  }

  /** Returns whether the arg is an expansion option. */
  public boolean isExpansionOption() {
    return getOptionExpansion().length > 0;
  }

  /** Returns whether the arg is an expansion option. */
  public boolean hasImplicitRequirements() {
    return (getImplicitRequirements().length > 0);
  }

  /**
   * For an option that does not use {@link Option#allowMultiple}, returns its type. For an option
   * that does use it, asserts that the type is a {@code List<T>} and returns its element type
   * {@code T}.
   */
  public Type getFieldSingularType() {
    Type type = getSingularType();
    if (allowsMultiple()) {
      // The validity of the converter is checked at compile time. We know the type to be
      // List<singularType>.
      ParameterizedType pfieldType = (ParameterizedType) type;
      type = pfieldType.getActualTypeArguments()[0];
    }
    return type;
  }

  protected abstract Type getSingularType();

  /** Returns the {@link Converter} that will be used for this option. */
  public Converter<?> getConverter() {
    if (converter != null) {
      return converter;
    }

    synchronized (this) {
      if (converter != null) {
        return converter;
      }

      @SuppressWarnings("rawtypes") // Converter itself has a type argument
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
              String.format("Error in the provided converter for option %s", getMemberName()), e);
        }
      }
      return converter;
    }
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

  /**
   * Returns whether an option requires a value when instantiated, or instead can be present without
   * an explicit value.
   */
  public boolean requiresValue() {
    return !isVoidField() && !usesBooleanValueSyntax();
  }

  /**
   * Wraps a converted default value into a {@link List} if the converter doesn't do it on its own.
   *
   * <p>This is to make sure multiple ({@link Option#allowMultiple()}) options' default values are
   * always converted to a list representation.
   */
  @SuppressWarnings("unchecked") // Not an unchecked cast - there's an explicit type check before it
  protected static List<Object> maybeWrapMultipleDefaultValue(Object convertedDefaultValue) {
    if (convertedDefaultValue instanceof List) {
      return (List<Object>) convertedDefaultValue;
    } else {
      return Arrays.asList(convertedDefaultValue);
    }
  }

  /** Returns the evaluated default value for this option. */
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
                getMemberName(), e.getMessage()),
            e);
      }

      return defaultValue;
    }
  }

  /** Returns the name of the member (field or method) that defines this option. */
  public abstract String getMemberName();

  @Override
  public int compareTo(OptionDefinition o) {
    return getOptionName().compareTo(o.getOptionName());
  }
}
