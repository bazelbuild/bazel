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

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.Lists;
import com.google.common.collect.Maps;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Collection;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.Immutable;

/**
 * An immutable selection of options data corresponding to a set of options
 * classes. The data is collected using reflection, which can be expensive.
 * Therefore this class can be used internally to cache the results.
 */
@Immutable
final class OptionsData extends OpaqueOptionsData {

  /**
   * These are the options-declaring classes which are annotated with
   * {@link Option} annotations.
   */
  private final Map<Class<? extends OptionsBase>, Constructor<?>> optionsClasses;

  /** Maps option name to Option-annotated Field. */
  private final Map<String, Field> nameToField;

  /** Maps option abbreviation to Option-annotated Field. */
  private final Map<Character, Field> abbrevToField;

  /**
   * For each options class, contains a list of all Option-annotated fields in
   * that class.
   */
  private final Map<Class<? extends OptionsBase>, List<Field>> allOptionsFields;

  /**
   * Mapping from each Option-annotated field to the default value for that
   * field.
   */
  private final Map<Field, Object> optionDefaults;

  /**
   * Mapping from each Option-annotated field to the proper converter.
   *
   * @see #findConverter
   */
  private final Map<Field, Converter<?>> converters;

  /**
   * Mapping from each Option-annotated field to a boolean for whether that field allows multiple
   * values.
   */
  private final Map<Field, Boolean> allowMultiple;

  private OptionsData(Map<Class<? extends OptionsBase>, Constructor<?>> optionsClasses,
                      Map<String, Field> nameToField,
                      Map<Character, Field> abbrevToField,
                      Map<Class<? extends OptionsBase>, List<Field>> allOptionsFields,
                      Map<Field, Object> optionDefaults,
                      Map<Field, Converter<?>> converters,
                      Map<Field, Boolean> allowMultiple) {
    this.optionsClasses = ImmutableMap.copyOf(optionsClasses);
    this.allOptionsFields = ImmutableMap.copyOf(allOptionsFields);
    this.nameToField = ImmutableMap.copyOf(nameToField);
    this.abbrevToField = ImmutableMap.copyOf(abbrevToField);
    // Can't use an ImmutableMap here because of null values.
    this.optionDefaults = Collections.unmodifiableMap(optionDefaults);
    this.converters = ImmutableMap.copyOf(converters);
    this.allowMultiple = ImmutableMap.copyOf(allowMultiple);
  }

  public Collection<Class<? extends OptionsBase>> getOptionsClasses() {
    return optionsClasses.keySet();
  }

  @SuppressWarnings("unchecked") // The construction ensures that the case is always valid.
  public <T extends OptionsBase> Constructor<T> getConstructor(Class<T> clazz) {
    return (Constructor<T>) optionsClasses.get(clazz);
  }

  public Field getFieldFromName(String name) {
    return nameToField.get(name);
  }

  public Iterable<Map.Entry<String, Field>> getAllNamedFields() {
    return nameToField.entrySet();
  }

  public Field getFieldForAbbrev(char abbrev) {
    return abbrevToField.get(abbrev);
  }

  public List<Field> getFieldsForClass(Class<? extends OptionsBase> optionsClass) {
    return allOptionsFields.get(optionsClass);
  }

  public Object getDefaultValue(Field field) {
    return optionDefaults.get(field);
  }

  public Converter<?> getConverter(Field field) {
    return converters.get(field);
  }

  public boolean getAllowMultiple(Field field) {
    return allowMultiple.get(field);
  }

  /**
   * For an option that does not use {@link Option#allowMultiple}, returns its type. For an option
   * that does use it, asserts that the type is a {@code List<T>} and returns its element type
   * {@code T}.
   */
  private static Type getFieldSingularType(Field field, Option annotation) {
    Type fieldType = field.getGenericType();
    if (annotation.allowMultiple()) {
      // If the type isn't a List<T>, this is an error in the option's declaration.
      if (!(fieldType instanceof ParameterizedType)) {
        throw new AssertionError("Type of multiple occurrence option must be a List<...>");
      }
      ParameterizedType pfieldType = (ParameterizedType) fieldType;
      if (pfieldType.getRawType() != List.class) {
        throw new AssertionError("Type of multiple occurrence option must be a List<...>");
      }
      fieldType = pfieldType.getActualTypeArguments()[0];
    }
    return fieldType;
  }

  /**
   * Returns whether a field should be considered as boolean.
   *
   * <p>Can be used for usage help and controlling whether the "no" prefix is allowed.
   */
  static boolean isBooleanField(Field field) {
    return field.getType().equals(boolean.class)
        || field.getType().equals(TriState.class)
        || findConverter(field) instanceof BoolOrEnumConverter;
  }

  /** Returns whether a field has Void type. */
  static boolean isVoidField(Field field) {
    return field.getType().equals(Void.class);
  }

  /**
   * Given an {@code @Option}-annotated field, retrieves the {@link Converter} that will be used,
   * taking into account the default converters if an explicit one is not specified.
   */
  static Converter<?> findConverter(Field optionField) {
    Option annotation = optionField.getAnnotation(Option.class);
    if (annotation.converter() == Converter.class) {
      // No converter provided, use the default one.
      Type type = getFieldSingularType(optionField, annotation);
      Converter<?> converter = Converters.DEFAULT_CONVERTERS.get(type);
      if (converter == null) {
        throw new AssertionError(
            "No converter found for "
                + type
                + "; possible fix: add "
                + "converter=... to @Option annotation for "
                + optionField.getName());
      }
      return converter;
    }
    try {
      // Instantiate the given Converter class.
      Class<?> converter = annotation.converter();
      Constructor<?> constructor = converter.getConstructor(new Class<?>[0]);
      return (Converter<?>) constructor.newInstance(new Object[0]);
    } catch (Exception e) {
      throw new AssertionError(e);
    }
  }

  private static List<Field> getAllAnnotatedFields(Class<? extends OptionsBase> optionsClass) {
    List<Field> allFields = Lists.newArrayList();
    for (Field field : optionsClass.getFields()) {
      if (field.isAnnotationPresent(Option.class)) {
        allFields.add(field);
      }
    }
    if (allFields.isEmpty()) {
      throw new IllegalStateException(optionsClass + " has no public @Option-annotated fields");
    }
    return ImmutableList.copyOf(allFields);
  }

  private static Object retrieveDefaultFromAnnotation(Field optionField) {
    Converter<?> converter = findConverter(optionField);
    String defaultValueAsString = OptionsParserImpl.getDefaultOptionString(optionField);
    // Special case for "null"
    if (OptionsParserImpl.isSpecialNullDefault(defaultValueAsString, optionField)) {
      return null;
    }
    boolean allowsMultiple = optionField.getAnnotation(Option.class).allowMultiple();
    // If the option allows multiple values then we intentionally return the empty list as
    // the default value of this option since it is not always the case that an option
    // that allows multiple values will have a converter that returns a list value.
    if (allowsMultiple) {
      return Collections.emptyList();
    }
    // Otherwise try to convert the default value using the converter
    Object convertedValue;
    try {
      convertedValue = converter.convert(defaultValueAsString);
    } catch (OptionsParsingException e) {
      throw new IllegalStateException("OptionsParsingException while "
          + "retrieving default for " + optionField.getName() + ": "
          + e.getMessage());
    }
    return convertedValue;
  }

  static OptionsData of(Collection<Class<? extends OptionsBase>> classes) {
    Map<Class<? extends OptionsBase>, Constructor<?>> constructorBuilder = Maps.newHashMap();
    Map<Class<? extends OptionsBase>, List<Field>> allOptionsFieldsBuilder = Maps.newHashMap();
    Map<String, Field> nameToFieldBuilder = Maps.newHashMap();
    Map<Character, Field> abbrevToFieldBuilder = Maps.newHashMap();
    Map<Field, Object> optionDefaultsBuilder = Maps.newHashMap();
    Map<Field, Converter<?>> convertersBuilder = Maps.newHashMap();
    Map<Field, Boolean> allowMultipleBuilder = Maps.newHashMap();

    // Read all Option annotations:
    for (Class<? extends OptionsBase> parsedOptionsClass : classes) {
      try {
        Constructor<? extends OptionsBase> constructor =
            parsedOptionsClass.getConstructor(new Class[0]);
        constructorBuilder.put(parsedOptionsClass, constructor);
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(parsedOptionsClass
            + " lacks an accessible default constructor");
      }
      List<Field> fields = getAllAnnotatedFields(parsedOptionsClass);
      allOptionsFieldsBuilder.put(parsedOptionsClass, fields);

      for (Field field : fields) {
        Option annotation = field.getAnnotation(Option.class);

        Type fieldType = getFieldSingularType(field, annotation);

        // Get the converter return type.
        @SuppressWarnings("rawtypes")
        Class<? extends Converter> converter = annotation.converter();
        if (converter == Converter.class) {
          Converter<?> actualConverter = Converters.DEFAULT_CONVERTERS.get(fieldType);
          if (actualConverter == null) {
            throw new AssertionError("Cannot find converter for field of type "
                + field.getType() + " named " + field.getName()
                + " in class " + field.getDeclaringClass().getName());
          }
          converter = actualConverter.getClass();
        }
        if (Modifier.isAbstract(converter.getModifiers())) {
          throw new AssertionError("The converter type (" + converter
              + ") must be a concrete type");
        }
        Type converterResultType;
        try {
          Method convertMethod = converter.getMethod("convert", String.class);
          converterResultType = GenericTypeHelper.getActualReturnType(converter, convertMethod);
        } catch (NoSuchMethodException e) {
          throw new AssertionError("A known converter object doesn't implement the convert"
              + " method");
        }

        if (annotation.allowMultiple()) {
          if (GenericTypeHelper.getRawType(converterResultType) == List.class) {
            Type elementType =
                ((ParameterizedType) converterResultType).getActualTypeArguments()[0];
            if (!GenericTypeHelper.isAssignableFrom(fieldType, elementType)) {
              throw new AssertionError("If the converter return type of a multiple occurance " +
                  "option is a list, then the type of list elements (" + fieldType + ") must be " +
                  "assignable from the converter list element type (" + elementType + ")");
            }
          } else {
            if (!GenericTypeHelper.isAssignableFrom(fieldType, converterResultType)) {
              throw new AssertionError("Type of list elements (" + fieldType +
                  ") for multiple occurrence option must be assignable from the converter " +
                  "return type (" + converterResultType + ")");
            }
          }
        } else {
          if (!GenericTypeHelper.isAssignableFrom(fieldType, converterResultType)) {
            throw new AssertionError("Type of field (" + fieldType +
                ") must be assignable from the converter " +
                "return type (" + converterResultType + ")");
          }
        }

        if (annotation.name() == null) {
          throw new AssertionError(
              "Option cannot have a null name");
        }
        if (nameToFieldBuilder.put(annotation.name(), field) != null) {
          throw new DuplicateOptionDeclarationException(
              "Duplicate option name: --" + annotation.name());
        }
        if (!annotation.oldName().isEmpty()) {
          if (nameToFieldBuilder.put(annotation.oldName(), field) != null) {
            throw new DuplicateOptionDeclarationException(
                "Old option name duplicates option name: --" + annotation.oldName());
          }
        }
        if (annotation.abbrev() != '\0') {
          if (abbrevToFieldBuilder.put(annotation.abbrev(), field) != null) {
            throw new DuplicateOptionDeclarationException(
                  "Duplicate option abbrev: -" + annotation.abbrev());
          }
        }

        optionDefaultsBuilder.put(field, retrieveDefaultFromAnnotation(field));

        convertersBuilder.put(field, findConverter(field));

        allowMultipleBuilder.put(field, annotation.allowMultiple());
      }
    }
    return new OptionsData(constructorBuilder, nameToFieldBuilder, abbrevToFieldBuilder,
        allOptionsFieldsBuilder, optionDefaultsBuilder, convertersBuilder, allowMultipleBuilder);
  }
}
