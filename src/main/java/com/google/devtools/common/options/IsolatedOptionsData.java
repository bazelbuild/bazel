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
import com.google.devtools.common.options.OptionDefinition.NotAnOptionException;
import com.google.devtools.common.options.OptionsParser.ConstructionException;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.lang.reflect.ParameterizedType;
import java.lang.reflect.Type;
import java.util.Arrays;
import java.util.Collection;
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
      Map<Class<? extends OptionsBase>, Boolean> usesOnlyCoreTypes) {
    this.optionsClasses = ImmutableMap.copyOf(optionsClasses);
    this.nameToField = ImmutableMap.copyOf(nameToField);
    this.abbrevToField = ImmutableMap.copyOf(abbrevToField);
    this.allOptionsFields = ImmutableMap.copyOf(allOptionsFields);
    this.usesOnlyCoreTypes = ImmutableMap.copyOf(usesOnlyCoreTypes);
  }

  protected IsolatedOptionsData(IsolatedOptionsData other) {
    this(
        other.optionsClasses,
        other.nameToField,
        other.abbrevToField,
        other.allOptionsFields,
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

  public boolean getUsesOnlyCoreTypes(Class<? extends OptionsBase> optionsClass) {
    return usesOnlyCoreTypes.get(optionsClass);
  }

  /** Returns all {@code optionDefinitions}, ordered by their option name (not their field name). */
  private static ImmutableList<OptionDefinition> getAllOptionDefinitionsSorted(
      Class<? extends OptionsBase> optionsClass) {
    return Arrays.stream(optionsClass.getFields())
        .map(
            field -> {
              try {
                return OptionDefinition.extractOptionDefinition(field);
              } catch (NotAnOptionException e) {
                // Ignore non-@Option annotated fields. Requiring all fields in the OptionsBase to
                // be @Option-annotated requires a depot cleanup.
                return null;
              }
            })
        .filter(Objects::nonNull)
        .sorted(OptionDefinition.BY_OPTION_NAME)
        .collect(ImmutableList.toImmutableList());
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

        Type fieldType = optionDefinition.getFieldSingularType();
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

        if (optionDefinition.isBooleanField()) {
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
          if (optionDefinition.isBooleanField()) {
            checkAndUpdateBooleanAliases(nameToFieldBuilder, booleanAliasMap, oldName);
          }
        }
        if (optionDefinition.getAbbreviation() != '\0') {
          checkForCollisions(
              abbrevToFieldBuilder, optionDefinition.getAbbreviation(), "option abbreviation");
          abbrevToFieldBuilder.put(optionDefinition.getAbbreviation(), optionDefinition);
        }
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
        usesOnlyCoreTypesBuilder);
  }

}
