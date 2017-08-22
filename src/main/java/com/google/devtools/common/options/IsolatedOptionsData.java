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
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
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
   * Mapping from option name to {@code OptionDefinition}. Entries appear ordered first by their
   * options class (the order in which they were passed to {@link #from(Collection)}, and then in
   * alphabetic order within each options class.
   */
  private final ImmutableMap<String, OptionDefinition> nameToField;

  /** Mapping from option abbreviation to {@code OptionDefinition} (unordered). */
  private final ImmutableMap<Character, OptionDefinition> abbrevToField;

  /**
   * Mapping from options class to a list of all {@code OptionFields} in that class. The map entries
   * are unordered, but the fields in the lists are ordered alphabetically.
   */
  private final ImmutableMap<Class<? extends OptionsBase>, ImmutableList<OptionDefinition>>
      allOptionsFields;

  /**
   * Mapping from each {@code Option}-annotated field to the default value for that field
   * (unordered).
   *
   * <p>(This is immutable like the others, but uses {@code Collections.unmodifiableMap} to support
   * null values.)
   */
  private final Map<OptionDefinition, Object> optionDefaults;

  /**
   * Mapping from each {@code Option}-annotated field to the proper converter (unordered).
   *
   * @see #findConverter
   */
  private final ImmutableMap<OptionDefinition, Converter<?>> converters;

  /**
   * Mapping from each {@code Option}-annotated field to a boolean for whether that field allows
   * multiple values (unordered).
   */
  private final ImmutableMap<OptionDefinition, Boolean> allowMultiple;

  /**
   * Mapping from each options class to whether or not it has the {@link UsesOnlyCoreTypes}
   * annotation (unordered).
   */
  private final ImmutableMap<Class<? extends OptionsBase>, Boolean> usesOnlyCoreTypes;

  /** These categories used to indicate OptionUsageRestrictions, but no longer. */
  private static final ImmutableList<String> DEPRECATED_CATEGORIES = ImmutableList.of(
      "undocumented", "hidden", "internal");

  private IsolatedOptionsData(
      Map<Class<? extends OptionsBase>, Constructor<?>> optionsClasses,
      Map<String, OptionDefinition> nameToField,
      Map<Character, OptionDefinition> abbrevToField,
      Map<Class<? extends OptionsBase>, ImmutableList<OptionDefinition>> allOptionsFields,
      Map<OptionDefinition, Object> optionDefaults,
      Map<OptionDefinition, Converter<?>> converters,
      Map<OptionDefinition, Boolean> allowMultiple,
      Map<Class<? extends OptionsBase>, Boolean> usesOnlyCoreTypes) {
    this.optionsClasses = ImmutableMap.copyOf(optionsClasses);
    this.nameToField = ImmutableMap.copyOf(nameToField);
    this.abbrevToField = ImmutableMap.copyOf(abbrevToField);
    this.allOptionsFields = ImmutableMap.copyOf(allOptionsFields);
    // Can't use an ImmutableMap here because of null values.
    this.optionDefaults = Collections.unmodifiableMap(optionDefaults);
    this.converters = ImmutableMap.copyOf(converters);
    this.allowMultiple = ImmutableMap.copyOf(allowMultiple);
    this.usesOnlyCoreTypes = ImmutableMap.copyOf(usesOnlyCoreTypes);
  }

  protected IsolatedOptionsData(IsolatedOptionsData other) {
    this(
        other.optionsClasses,
        other.nameToField,
        other.abbrevToField,
        other.allOptionsFields,
        other.optionDefaults,
        other.converters,
        other.allowMultiple,
        other.usesOnlyCoreTypes);
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

  public OptionDefinition getFieldFromName(String name) {
    return nameToField.get(name);
  }

  /**
   * Returns all pairs of option names (not field names) and their corresponding {@link Field}
   * objects. Entries appear ordered first by their options class (the order in which they were
   * passed to {@link #from(Collection)}, and then in alphabetic order within each options class.
   */
  public Iterable<Map.Entry<String, OptionDefinition>> getAllNamedFields() {
    return nameToField.entrySet();
  }

  public OptionDefinition getFieldForAbbrev(char abbrev) {
    return abbrevToField.get(abbrev);
  }

  /**
   * Returns a list of all {@link Field} objects for options in the given options class, ordered
   * alphabetically by option name.
   */
  public ImmutableList<OptionDefinition> getOptionDefinitionsFromClass(
      Class<? extends OptionsBase> optionsClass) {
    return allOptionsFields.get(optionsClass);
  }

  public Object getDefaultValue(OptionDefinition optionDefinition) {
    return optionDefaults.get(optionDefinition);
  }

  public Converter<?> getConverter(OptionDefinition optionDefinition) {
    return converters.get(optionDefinition);
  }

  public boolean getAllowMultiple(OptionDefinition optionDefinition) {
    return allowMultiple.get(optionDefinition);
  }

  public boolean getUsesOnlyCoreTypes(Class<? extends OptionsBase> optionsClass) {
    return usesOnlyCoreTypes.get(optionsClass);
  }

  /**
   * For an option that does not use {@link Option#allowMultiple}, returns its type. For an option
   * that does use it, asserts that the type is a {@code List<T>} and returns its element type
   * {@code T}.
   */
  private static Type getFieldSingularType(OptionDefinition optionDefinition) {
    Type fieldType = optionDefinition.getField().getGenericType();
    if (optionDefinition.allowsMultiple()) {
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
  boolean isBooleanField(OptionDefinition optionDefinition) {
    return isBooleanField(optionDefinition, getConverter(optionDefinition));
  }

  private static boolean isBooleanField(OptionDefinition optionDefinition, Converter<?> converter) {
    return optionDefinition.getType().equals(boolean.class)
        || optionDefinition.getType().equals(TriState.class)
        || converter instanceof BoolOrEnumConverter;
  }

  /**
   * Given an {@code @Option}-annotated field, retrieves the {@link Converter} that will be used,
   * taking into account the default converters if an explicit one is not specified.
   */
  private static Converter<?> findConverter(OptionDefinition optionDefinition) {
    if (optionDefinition.getProvidedConverter() == Converter.class) {
      // No converter provided, use the default one.
      Type type = getFieldSingularType(optionDefinition);
      Converter<?> converter = Converters.DEFAULT_CONVERTERS.get(type);
      if (converter == null) {
        throw new ConstructionException(
            "No converter found for "
                + type
                + "; possible fix: add "
                + "converter=... to @Option annotation for "
                + optionDefinition.getField().getName());
      }
      return converter;
    }
    try {
      // Instantiate the given Converter class.
      Class<?> converter = optionDefinition.getProvidedConverter();
      Constructor<?> constructor = converter.getConstructor();
      return (Converter<?>) constructor.newInstance();
    } catch (Exception e) {
      // This indicates an error in the Converter, and should be discovered the first time it is
      // used.
      throw new ConstructionException(e);
    }
  }
  /** Returns all {@code optionDefinitions}, ordered by their option name (not their field name). */
  private static ImmutableList<OptionDefinition> getAllOptionDefinitionsSorted(
      Class<? extends OptionsBase> optionsClass) {
    return Arrays.stream(optionsClass.getFields())
        .map(field -> {
          try {
            return OptionDefinition.extractOptionDefinition(field);
          } catch (ConstructionException e) {
            // Ignore non-@Option annotated fields. Requiring all fields in the OptionsBase to be
            // @Option-annotated requires a depot cleanup.
            return null;
          }
        })
        .filter(Objects::nonNull)
        .sorted(OptionDefinition.BY_OPTION_NAME)
        .collect(ImmutableList.toImmutableList());
  }

  private static Object retrieveDefaultValue(OptionDefinition optionDefinition) {
    Converter<?> converter = findConverter(optionDefinition);
    String defaultValueAsString = optionDefinition.getUnparsedDefaultValue();
    // Special case for "null"
    if (optionDefinition.isSpecialNullDefault()) {
      return null;
    }
    boolean allowsMultiple = optionDefinition.allowsMultiple();
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
      throw new IllegalStateException(
          "OptionsParsingException while "
              + "retrieving default for "
              + optionDefinition.getField().getName()
              + ": "
              + e.getMessage());
    }
    return convertedValue;
  }

  private static <A> void checkForCollisions(
      Map<A, OptionDefinition> aFieldMap, A optionName, String description) {
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
      Map<String, OptionDefinition> nameToFieldMap,
      Map<String, String> booleanAliasMap,
      String optionName) {
    // Check that the negating alias does not conflict with existing flags.
    checkForCollisions(nameToFieldMap, "no" + optionName, "boolean option alias");

    // Record that the boolean option takes up additional namespace for its negating alias.
    booleanAliasMap.put("no" + optionName, optionName);
  }

  private static void checkEffectTagRationality(String optionName, OptionEffectTag[] effectTags) {
    // Check that there is at least one OptionEffectTag listed.
    if (effectTags.length < 1) {
      throw new ConstructionException(
          "Option "
              + optionName
              + " does not list at least one OptionEffectTag. If the option has no effect, "
              + "please add NO_OP, otherwise, add a tag representing its effect.");
    } else if (effectTags.length > 1) {
      // If there are more than 1 tag, make sure that NO_OP and UNKNOWN is not one of them.
      // These don't make sense if other effects are listed.
      ImmutableList<OptionEffectTag> tags = ImmutableList.copyOf(effectTags);
      if (tags.contains(OptionEffectTag.UNKNOWN)) {
        throw new ConstructionException(
            "Option "
                + optionName
                + " includes UNKNOWN with other, known, effects. Please remove UNKNOWN from "
                + "the list.");
      }
      if (tags.contains(OptionEffectTag.NO_OP)) {
        throw new ConstructionException(
            "Option "
                + optionName
                + " includes NO_OP with other effects. This doesn't make much sense. Please "
                + "remove NO_OP or the actual effects from the list, whichever is correct.");
      }
    }
  }

  private static void checkMetadataTagAndCategoryRationality(
      String optionName, OptionMetadataTag[] metadataTags, OptionDocumentationCategory category) {
    for (OptionMetadataTag tag : metadataTags) {
      if (tag == OptionMetadataTag.HIDDEN || tag == OptionMetadataTag.INTERNAL) {
        if (category != OptionDocumentationCategory.UNDOCUMENTED) {
          throw new ConstructionException(
              "Option "
                  + optionName
                  + " has metadata tag "
                  + tag
                  + " but does not have category UNDOCUMENTED. "
                  + "Please fix.");
        }
      }
    }
  }

  /**
   * Constructs an {@link IsolatedOptionsData} object for a parser that knows about the given
   * {@link OptionsBase} classes. No inter-option analysis is done. Performs basic sanity checking
   * on each option in isolation.
   */
  static IsolatedOptionsData from(Collection<Class<? extends OptionsBase>> classes) {
    // Mind which fields have to preserve order.
    Map<Class<? extends OptionsBase>, Constructor<?>> constructorBuilder = new LinkedHashMap<>();
    Map<Class<? extends OptionsBase>, ImmutableList<OptionDefinition>> allOptionsFieldsBuilder =
        new HashMap<>();
    Map<String, OptionDefinition> nameToFieldBuilder = new LinkedHashMap<>();
    Map<Character, OptionDefinition> abbrevToFieldBuilder = new HashMap<>();
    Map<OptionDefinition, Object> optionDefaultsBuilder = new HashMap<>();
    Map<OptionDefinition, Converter<?>> convertersBuilder = new HashMap<>();
    Map<OptionDefinition, Boolean> allowMultipleBuilder = new HashMap<>();

    // Maps the negated boolean flag aliases to the original option name.
    Map<String, String> booleanAliasMap = new HashMap<>();

    Map<Class<? extends OptionsBase>, Boolean> usesOnlyCoreTypesBuilder = new HashMap<>();

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
      ImmutableList<OptionDefinition> optionDefinitions =
          getAllOptionDefinitionsSorted(parsedOptionsClass);
      allOptionsFieldsBuilder.put(parsedOptionsClass, optionDefinitions);

      for (OptionDefinition optionDefinition : optionDefinitions) {
        String optionName = optionDefinition.getOptionName();

        // Check that the option makes sense on its own, as defined.
        if (optionName == null) {
          throw new ConstructionException("Option cannot have a null name");
        }

        if (DEPRECATED_CATEGORIES.contains(optionDefinition.getOptionCategory())) {
          throw new ConstructionException(
              "Documentation level is no longer read from the option category. Category \""
                  + optionDefinition.getOptionCategory()
                  + "\" in option \""
                  + optionName
                  + "\" is disallowed.");
        }

        checkEffectTagRationality(optionName, optionDefinition.getOptionEffectTags());
        checkMetadataTagAndCategoryRationality(
            optionName,
            optionDefinition.getOptionMetadataTags(),
            optionDefinition.getDocumentationCategory());
        Type fieldType = getFieldSingularType(optionDefinition);
        // For simple, static expansions, don't accept non-Void types.
        if (optionDefinition.getOptionExpansion().length != 0 && !optionDefinition.isVoidField()) {
          throw new ConstructionException(
              "Option "
                  + optionDefinition.getOptionName()
                  + " is an expansion flag with a static expansion, but does not have Void type.");
        }

        // Get the converter's return type to check that it matches the option type.
        @SuppressWarnings("rawtypes")
        Class<? extends Converter> converterClass = optionDefinition.getProvidedConverter();
        if (converterClass == Converter.class) {
          Converter<?> actualConverter = Converters.DEFAULT_CONVERTERS.get(fieldType);
          if (actualConverter == null) {
            throw new ConstructionException(
                "Cannot find converter for field of type "
                    + optionDefinition.getType()
                    + " named "
                    + optionDefinition.getField().getName()
                    + " in class "
                    + optionDefinition.getField().getDeclaringClass().getName());
          }
          converterClass = actualConverter.getClass();
        }
        if (Modifier.isAbstract(converterClass.getModifiers())) {
          throw new ConstructionException(
              "The converter type " + converterClass + " must be a concrete type");
        }
        Type converterResultType;
        try {
          Method convertMethod = converterClass.getMethod("convert", String.class);
          converterResultType =
              GenericTypeHelper.getActualReturnType(converterClass, convertMethod);
        } catch (NoSuchMethodException e) {
          throw new ConstructionException(
              "A known converter object doesn't implement the convert method");
        }
        Converter<?> converter = findConverter(optionDefinition);
        convertersBuilder.put(optionDefinition, converter);

        if (optionDefinition.allowsMultiple()) {
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

        if (isBooleanField(optionDefinition, converter)) {
          checkAndUpdateBooleanAliases(nameToFieldBuilder, booleanAliasMap, optionName);
        }

        checkForCollisions(nameToFieldBuilder, optionName, "option");
        checkForBooleanAliasCollisions(booleanAliasMap, optionName, "option");
        nameToFieldBuilder.put(optionName, optionDefinition);

        if (!optionDefinition.getOldOptionName().isEmpty()) {
          String oldName = optionDefinition.getOldOptionName();
          checkForCollisions(nameToFieldBuilder, oldName, "old option name");
          checkForBooleanAliasCollisions(booleanAliasMap, oldName, "old option name");
          nameToFieldBuilder.put(optionDefinition.getOldOptionName(), optionDefinition);

          // If boolean, repeat the alias dance for the old name.
          if (isBooleanField(optionDefinition, converter)) {
            checkAndUpdateBooleanAliases(nameToFieldBuilder, booleanAliasMap, oldName);
          }
        }
        if (optionDefinition.getAbbreviation() != '\0') {
          checkForCollisions(
              abbrevToFieldBuilder, optionDefinition.getAbbreviation(), "option abbreviation");
          abbrevToFieldBuilder.put(optionDefinition.getAbbreviation(), optionDefinition);
        }

        optionDefaultsBuilder.put(optionDefinition, retrieveDefaultValue(optionDefinition));
        allowMultipleBuilder.put(optionDefinition, optionDefinition.allowsMultiple());
      }

      boolean usesOnlyCoreTypes = parsedOptionsClass.isAnnotationPresent(UsesOnlyCoreTypes.class);
      if (usesOnlyCoreTypes) {
        // Validate that @UsesOnlyCoreTypes was used correctly.
        for (OptionDefinition optionDefinition : optionDefinitions) {
          // The classes in coreTypes are all final. But even if they weren't, we only want to check
          // for exact matches; subclasses would not be considered core types.
          if (!UsesOnlyCoreTypes.CORE_TYPES.contains(optionDefinition.getType())) {
            throw new ConstructionException(
                "Options class '"
                    + parsedOptionsClass.getName()
                    + "' is marked as "
                    + "@UsesOnlyCoreTypes, but field '"
                    + optionDefinition.getField().getName()
                    + "' has type '"
                    + optionDefinition.getType().getName()
                    + "'");
          }
        }
      }
      usesOnlyCoreTypesBuilder.put(parsedOptionsClass, usesOnlyCoreTypes);
    }

    return new IsolatedOptionsData(
        constructorBuilder,
        nameToFieldBuilder,
        abbrevToFieldBuilder,
        allOptionsFieldsBuilder,
        optionDefaultsBuilder,
        convertersBuilder,
        allowMultipleBuilder,
        usesOnlyCoreTypesBuilder);
  }

}
