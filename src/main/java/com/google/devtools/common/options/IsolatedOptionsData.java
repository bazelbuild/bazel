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
import com.google.common.collect.Ordering;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.concurrent.Immutable;

/**
 * A selection of options data corresponding to a set of {@link OptionsBase} subclasses (options
 * classes). The data is collected using reflection, which can be expensive. Therefore this class
 * can be used internally to cache the results.
 *
 * <p>The data is isolated in the sense that it has not yet been processed to add
 * inter-option-dependent information -- namely, the results of evaluating expansion functions. The
 * {@link OptionsData} subclass stores this added information. The reason for the split is so that
 * we can avoid exposing to expansion functions the effects of evaluating other expansion functions,
 * to ensure that the order in which they run is not significant.
 *
 * <p>This class is immutable so long as the converters and default values associated with the
 * options are immutable.
 */
@Immutable
public class IsolatedOptionsData extends OpaqueOptionsData {

  /**
   * Mapping from each options class to its no-arg constructor. Entries appear in the same order
   * that they were passed to {@link #from(Collection)}.
   */
  private final ImmutableMap<Class<? extends OptionsBase>, Constructor<?>> optionsClasses;

  /**
   * Mapping from option name to {@code @Option}-annotated field. Entries appear ordered first by
   * their options class (the order in which they were passed to {@link #from(Collection)}, and then
   * in alphabetic order within each options class.
   */
  private final ImmutableMap<String, Field> nameToField;

  /** Mapping from option abbreviation to {@code Option}-annotated field (unordered). */
  private final ImmutableMap<Character, Field> abbrevToField;

  /**
   * Mapping from options class to a list of all {@code Option}-annotated fields in that class. The
   * map entries are unordered, but the fields in the lists are ordered alphabetically.
   */
  private final ImmutableMap<Class<? extends OptionsBase>, ImmutableList<Field>> allOptionsFields;

  /**
   * Mapping from each {@code Option}-annotated field to the default value for that field
   * (unordered).
   *
   * <p>(This is immutable like the others, but uses {@code Collections.unmodifiableMap} to support
   * null values.)
   */
  private final Map<Field, Object> optionDefaults;

  /**
   * Mapping from each {@code Option}-annotated field to the proper converter (unordered).
   *
   * @see #findConverter
   */
  private final ImmutableMap<Field, Converter<?>> converters;

  /**
   * Mapping from each {@code Option}-annotated field to a boolean for whether that field allows
   * multiple values (unordered).
   */
  private final ImmutableMap<Field, Boolean> allowMultiple;

  /** These categories used to indicate OptionUsageRestrictions, but no longer. */
  private static final ImmutableList<String> DEPRECATED_CATEGORIES = ImmutableList.of(
      "undocumented", "hidden", "internal");

  private IsolatedOptionsData(
      Map<Class<? extends OptionsBase>,
      Constructor<?>> optionsClasses,
      Map<String, Field> nameToField,
      Map<Character, Field> abbrevToField,
      Map<Class<? extends OptionsBase>, ImmutableList<Field>> allOptionsFields,
      Map<Field, Object> optionDefaults,
      Map<Field, Converter<?>> converters,
      Map<Field, Boolean> allowMultiple) {
    this.optionsClasses = ImmutableMap.copyOf(optionsClasses);
    this.nameToField = ImmutableMap.copyOf(nameToField);
    this.abbrevToField = ImmutableMap.copyOf(abbrevToField);
    this.allOptionsFields = ImmutableMap.copyOf(allOptionsFields);
    // Can't use an ImmutableMap here because of null values.
    this.optionDefaults = Collections.unmodifiableMap(optionDefaults);
    this.converters = ImmutableMap.copyOf(converters);
    this.allowMultiple = ImmutableMap.copyOf(allowMultiple);
  }

  protected IsolatedOptionsData(IsolatedOptionsData other) {
    this(
        other.optionsClasses,
        other.nameToField,
        other.abbrevToField,
        other.allOptionsFields,
        other.optionDefaults,
        other.converters,
        other.allowMultiple);
  }

  /**
   * Returns all options classes indexed by this options data object, in the order they were passed
   * to {@link #from(Collection)}.
   */
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

  /**
   * Returns all pairs of option names (not field names) and their corresponding {@link Field}
   * objects. Entries appear ordered first by their options class (the order in which they were
   * passed to {@link #from(Collection)}, and then in alphabetic order within each options class.
   */
  public Iterable<Map.Entry<String, Field>> getAllNamedFields() {
    return nameToField.entrySet();
  }

  public Field getFieldForAbbrev(char abbrev) {
    return abbrevToField.get(abbrev);
  }

  /**
   * Returns a list of all {@link Field} objects for options in the given options class, ordered
   * alphabetically by option name.
   */
  public ImmutableList<Field> getFieldsForClass(Class<? extends OptionsBase> optionsClass) {
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
        throw new ConstructionException("Type of multiple occurrence option must be a List<...>");
      }
      ParameterizedType pfieldType = (ParameterizedType) fieldType;
      if (pfieldType.getRawType() != List.class) {
        throw new ConstructionException("Type of multiple occurrence option must be a List<...>");
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

  /** Returns whether the arg is an expansion option. */
  public static boolean isExpansionOption(Option annotation) {
    return (annotation.expansion().length > 0 || OptionsData.usesExpansionFunction(annotation));
  }

  /**
   * Returns whether the arg is an expansion option defined by an expansion function (and not a
   * constant expansion value).
   */
  static boolean usesExpansionFunction(Option annotation) {
    return annotation.expansionFunction() != ExpansionFunction.class;
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
        throw new ConstructionException(
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
      Constructor<?> constructor = converter.getConstructor();
      return (Converter<?>) constructor.newInstance();
    } catch (Exception e) {
      // This indicates an error in the Converter, and should be discovered the first time it is
      // used.
      throw new ConstructionException(e);
    }
  }

  private static final Ordering<Field> fieldOrdering =
      new Ordering<Field>() {
    @Override
    public int compare(Field f1, Field f2) {
      String n1 = f1.getAnnotation(Option.class).name();
      String n2 = f2.getAnnotation(Option.class).name();
      return n1.compareTo(n2);
    }
  };

  /**
   * Return all {@code @Option}-annotated fields, alphabetically ordered by their option name (not
   * their field name).
   */
  private static ImmutableList<Field> getAllAnnotatedFieldsSorted(
      Class<? extends OptionsBase> optionsClass) {
    List<Field> unsortedFields = new ArrayList<>();
    for (Field field : optionsClass.getFields()) {
      if (field.isAnnotationPresent(Option.class)) {
        unsortedFields.add(field);
      }
    }
    return fieldOrdering.immutableSortedCopy(unsortedFields);
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

  private static <A> void checkForCollisions(
      Map<A, Field> aFieldMap,
      A optionName,
      String description) {
    if (aFieldMap.containsKey(optionName)) {
      throw new DuplicateOptionDeclarationException(
          "Duplicate option name, due to " + description + ": --" + optionName);
    }
  }

  private static void checkForBooleanAliasCollisions(
      Map<String, String> booleanAliasMap,
      String optionName,
      String description) {
    if (booleanAliasMap.containsKey(optionName)) {
      throw new DuplicateOptionDeclarationException(
          "Duplicate option name, due to "
              + description
              + " --"
              + optionName
              + ", it conflicts with a negating alias for boolean flag --"
              + booleanAliasMap.get(optionName));
    }
  }

  private static void checkAndUpdateBooleanAliases(
      Map<String, Field> nameToFieldMap,
      Map<String, String> booleanAliasMap,
      String optionName) {
    // Check that the negating alias does not conflict with existing flags.
    checkForCollisions(nameToFieldMap, "no" + optionName, "boolean option alias");

    // Record that the boolean option takes up additional namespace for its negating alias.
    booleanAliasMap.put("no" + optionName, optionName);
  }

  /**
   * Constructs an {@link IsolatedOptionsData} object for a parser that knows about the given
   * {@link OptionsBase} classes. No inter-option analysis is done. Performs basic sanity checking
   * on each option in isolation.
   */
  static IsolatedOptionsData from(Collection<Class<? extends OptionsBase>> classes) {
    // Mind which fields have to preserve order.
    Map<Class<? extends OptionsBase>, Constructor<?>> constructorBuilder = new LinkedHashMap<>();
    Map<Class<? extends OptionsBase>, ImmutableList<Field>> allOptionsFieldsBuilder =
        new HashMap<>();
    Map<String, Field> nameToFieldBuilder = new LinkedHashMap<>();
    Map<Character, Field> abbrevToFieldBuilder = new HashMap<>();
    Map<Field, Object> optionDefaultsBuilder = new HashMap<>();
    Map<Field, Converter<?>> convertersBuilder = new HashMap<>();
    Map<Field, Boolean> allowMultipleBuilder = new HashMap<>();

    // Maps the negated boolean flag aliases to the original option name.
    Map<String, String> booleanAliasMap = new HashMap<>();

    // Read all Option annotations:
    for (Class<? extends OptionsBase> parsedOptionsClass : classes) {
      try {
        Constructor<? extends OptionsBase> constructor =
            parsedOptionsClass.getConstructor();
        constructorBuilder.put(parsedOptionsClass, constructor);
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(parsedOptionsClass
            + " lacks an accessible default constructor");
      }
      ImmutableList<Field> fields = getAllAnnotatedFieldsSorted(parsedOptionsClass);
      allOptionsFieldsBuilder.put(parsedOptionsClass, fields);

      for (Field field : fields) {
        Option annotation = field.getAnnotation(Option.class);
        String optionName = annotation.name();
        if (optionName == null) {
          throw new ConstructionException("Option cannot have a null name");
        }

        if (DEPRECATED_CATEGORIES.contains(annotation.category())) {
          throw new ConstructionException(
              "Documentation level is no longer read from the option category. Category \""
                  + annotation.category() + "\" in option \"" + optionName + "\" is disallowed.");
        }

        Type fieldType = getFieldSingularType(field, annotation);

        // Get the converter return type.
        @SuppressWarnings("rawtypes")
        Class<? extends Converter> converter = annotation.converter();
        if (converter == Converter.class) {
          Converter<?> actualConverter = Converters.DEFAULT_CONVERTERS.get(fieldType);
          if (actualConverter == null) {
            throw new ConstructionException("Cannot find converter for field of type "
                + field.getType() + " named " + field.getName()
                + " in class " + field.getDeclaringClass().getName());
          }
          converter = actualConverter.getClass();
        }
        if (Modifier.isAbstract(converter.getModifiers())) {
          throw new ConstructionException("The converter type " + converter
              + " must be a concrete type");
        }
        Type converterResultType;
        try {
          Method convertMethod = converter.getMethod("convert", String.class);
          converterResultType = GenericTypeHelper.getActualReturnType(converter, convertMethod);
        } catch (NoSuchMethodException e) {
          throw new ConstructionException(
              "A known converter object doesn't implement the convert method");
        }

        if (annotation.allowMultiple()) {
          if (GenericTypeHelper.getRawType(converterResultType) == List.class) {
            Type elementType =
                ((ParameterizedType) converterResultType).getActualTypeArguments()[0];
            if (!GenericTypeHelper.isAssignableFrom(fieldType, elementType)) {
              throw new ConstructionException(
                  "If the converter return type of a multiple occurrence option is a list, then "
                      + "the type of list elements ("
                      + fieldType
                      + ") must be assignable from the converter list element type ("
                      + elementType
                      + ")");
            }
          } else {
            if (!GenericTypeHelper.isAssignableFrom(fieldType, converterResultType)) {
              throw new ConstructionException(
                  "Type of list elements ("
                      + fieldType
                      + ") for multiple occurrence option must be assignable from the converter "
                      + "return type ("
                      + converterResultType
                      + ")");
            }
          }
        } else {
          if (!GenericTypeHelper.isAssignableFrom(fieldType, converterResultType)) {
            throw new ConstructionException(
                "Type of field ("
                    + fieldType
                    + ") must be assignable from the converter return type ("
                    + converterResultType
                    + ")");
          }
        }

        if (isBooleanField(field)) {
          checkAndUpdateBooleanAliases(nameToFieldBuilder, booleanAliasMap, optionName);
        }

        checkForCollisions(nameToFieldBuilder, optionName, "option");
        checkForBooleanAliasCollisions(booleanAliasMap, optionName, "option");
        nameToFieldBuilder.put(optionName, field);

        if (!annotation.oldName().isEmpty()) {
          String oldName = annotation.oldName();
          checkForCollisions(nameToFieldBuilder, oldName, "old option name");
          checkForBooleanAliasCollisions(booleanAliasMap, oldName, "old option name");
          nameToFieldBuilder.put(annotation.oldName(), field);

          // If boolean, repeat the alias dance for the old name.
          if (isBooleanField(field)) {
            checkAndUpdateBooleanAliases(nameToFieldBuilder, booleanAliasMap, oldName);
          }
        }
        if (annotation.abbrev() != '\0') {
          checkForCollisions(abbrevToFieldBuilder, annotation.abbrev(), "option abbreviation");
          abbrevToFieldBuilder.put(annotation.abbrev(), field);
        }

        optionDefaultsBuilder.put(field, retrieveDefaultFromAnnotation(field));

        convertersBuilder.put(field, findConverter(field));

        allowMultipleBuilder.put(field, annotation.allowMultiple());
      }
    }

    return new IsolatedOptionsData(
        constructorBuilder,
        nameToFieldBuilder,
        abbrevToFieldBuilder,
        allOptionsFieldsBuilder,
        optionDefaultsBuilder,
        convertersBuilder,
        allowMultipleBuilder);
  }

}
