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

import com.google.common.base.Verify;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;
import com.google.common.collect.ImmutableSet;
import java.lang.reflect.Constructor;
import java.lang.reflect.Method;
import java.util.Collection;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
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
// TODO(b/159980134): Can this be folded into OptionsData?
@Immutable
public class IsolatedOptionsData extends OpaqueOptionsData {

  /**
   * A little class whose only virtue is that it has a constructor which can be used to mark cases
   * where it's ambiguous which subclass should be instantiated for a given options base class.
   */
  private static class AmbiguousClassMarker {
    public AmbiguousClassMarker() {
      throw new IllegalStateException();
    }
  }

  private static final Constructor<?> AMBIGUOUS_MARKER_CTOR;

  static {
    try {
      AMBIGUOUS_MARKER_CTOR = AmbiguousClassMarker.class.getConstructor();
    } catch (NoSuchMethodException e) {
      throw new IllegalStateException(e);
    }
  }

  /**
   * Cache for the options in an OptionsBase.
   *
   * <p>Mapping from options class to a list of all {@code OptionFields} in that class. The map
   * entries are unordered, but the fields in the lists are ordered alphabetically. This caches the
   * work of reflection done for the same {@code optionsBase} across multiple {@link OptionsData}
   * instances, and must be used through the thread safe {@link
   * #getAllOptionDefinitionsForClass(Class)}
   */
  private static final ConcurrentMap<Class<? extends OptionsBase>, ImmutableList<OptionDefinition>>
      allOptionsDefinitions = new ConcurrentHashMap<>();

  /** Returns all {@code optionDefinitions}, ordered by their option name (not their field name). */
  public static ImmutableList<OptionDefinition> getAllOptionDefinitionsForClass(
      Class<? extends OptionsBase> optionsClass) {
    return allOptionsDefinitions.computeIfAbsent(
        optionsClass,
        optionsBaseClass -> {
          ImmutableList.Builder<OptionDefinition> builder = ImmutableList.builder();
          Verify.verify(
              optionsBaseClass.isAnnotationPresent(OptionsClass.class),
              "Options class %s should be annotated with @OptionsClass",
              optionsBaseClass.getName());

          for (Method method : optionsBaseClass.getMethods()) {
            MethodOptionDefinition optionDefinition = MethodOptionDefinition.from(method);
            if (optionDefinition != null) {
              builder.add(optionDefinition);
            }
          }

          return ImmutableList.sortedCopyOf(OptionDefinition.BY_OPTION_NAME, builder.build());
        });
  }

  /**
   * Mapping from each options class to its no-arg constructor. Entries appear in the same order
   * that they were passed to {@link #from(Collection, boolean)}.
   */
  private final ImmutableMap<Class<? extends OptionsBase>, Constructor<?>> optionsClasses;

  /**
   * The list of options classes that were passed to the constructor. This is used to return the
   * options classes in the order they were provided, and avoids returning superclasses of
   * registered classes, which would lead to duplicate options in help messages.
   */
  private final ImmutableList<Class<? extends OptionsBase>> primaryOptionsClasses;

  /**
   * Mapping from option name to {@code OptionDefinition}. Entries appear ordered first by their
   * options class (the order in which they were passed to {@link #from(Collection, boolean)}, and
   * then in alphabetic order within each options class.
   */
  private final ImmutableMap<String, OptionDefinition> nameToField;

  /**
   * For options that have an "OldName", this is a mapping from old name to its corresponding {@code
   * OptionDefinition}. Entries appear ordered first by their options class (the order in which they
   * were passed to {@link #from(Collection, boolean)}, and then in alphabetic order within each
   * options class.
   */
  private final ImmutableMap<String, OptionDefinition> oldNameToField;

  /** Mapping from option abbreviation to {@code OptionDefinition} (unordered). */
  private final ImmutableMap<Character, OptionDefinition> abbrevToField;

  private IsolatedOptionsData(
      Map<Class<? extends OptionsBase>, Constructor<?>> optionsClasses,
      ImmutableList<Class<? extends OptionsBase>> primaryOptionsClasses,
      Map<String, OptionDefinition> nameToField,
      Map<String, OptionDefinition> oldNameToField,
      Map<Character, OptionDefinition> abbrevToField) {
    this.optionsClasses = ImmutableMap.copyOf(optionsClasses);
    this.primaryOptionsClasses = primaryOptionsClasses;
    this.nameToField = ImmutableMap.copyOf(nameToField);
    this.oldNameToField = ImmutableMap.copyOf(oldNameToField);
    this.abbrevToField = ImmutableMap.copyOf(abbrevToField);
  }

  protected IsolatedOptionsData(IsolatedOptionsData other) {
    this(
        other.optionsClasses,
        other.primaryOptionsClasses,
        other.nameToField,
        other.oldNameToField,
        other.abbrevToField);
  }

  /**
   * Returns all options classes indexed by this options data object, in the order they were passed
   * to {@link #from(Collection, boolean)}.
   */
  public Collection<Class<? extends OptionsBase>> getOptionsClasses() {
    return primaryOptionsClasses;
  }

  @SuppressWarnings("unchecked") // The construction ensures that the case is always valid.
  public <T extends OptionsBase> Constructor<T> getConstructor(Class<T> clazz) {
    return (Constructor<T>) optionsClasses.get(clazz);
  }

  /**
   * Returns the option in this parser by the provided name, or {@code null} if none is found. This
   * will match both the canonical name of an option, and any old name listed that we still accept.
   */
  public OptionDefinition getOptionDefinitionFromName(String name) {
    return nameToField.getOrDefault(name, oldNameToField.get(name));
  }

  /**
   * Returns all {@link OptionDefinition} objects loaded, mapped by their canonical names. Entries
   * appear ordered first by their options class (the order in which they were passed to {@link
   * #from(Collection, boolean)}, and then in alphabetic order within each options class.
   */
  public ImmutableSet<Map.Entry<String, OptionDefinition>> getAllOptionDefinitions() {
    return nameToField.entrySet();
  }

  public OptionDefinition getFieldForAbbrev(char abbrev) {
    return abbrevToField.get(abbrev);
  }

  /**
   * Generic method to check for collisions between the names we give options. Useful for checking
   * both single-character abbreviations and full names.
   */
  private static <A> void checkForCollisions(
      Map<A, OptionDefinition> aFieldMap,
      A optionName,
      OptionDefinition definition,
      String description,
      boolean allowDuplicatesParsingEquivalently)
      throws DuplicateOptionDeclarationException {
    if (aFieldMap.containsKey(optionName)) {
      OptionDefinition otherDefinition = aFieldMap.get(optionName);
      if (allowDuplicatesParsingEquivalently
          && OptionDefinition.equivalentForParsing(otherDefinition, definition)) {
        return;
      }
      throw new DuplicateOptionDeclarationException(
          "Duplicate option name, due to " + description + ": --" + optionName);
    }
  }

  /**
   * All options, even non-boolean ones, should check that they do not conflict with previously
   * loaded boolean options.
   */
  private static void checkForBooleanAliasCollisions(
      Map<String, String> booleanAliasMap, String optionName, String description)
      throws DuplicateOptionDeclarationException {
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

  /**
   * For an {@code option} of boolean type, this checks that the boolean alias does not conflict
   * with other names, and adds the boolean alias to a list so that future flags can find if they
   * conflict with a boolean alias..
   */
  private static void checkAndUpdateBooleanAliases(
      Map<String, OptionDefinition> nameToFieldMap,
      Map<String, OptionDefinition> oldNameToFieldMap,
      Map<String, String> booleanAliasMap,
      String optionName,
      OptionDefinition optionDefinition,
      boolean allowDuplicatesParsingEquivalently)
      throws DuplicateOptionDeclarationException {
    // Check that the negating alias does not conflict with existing flags.
    checkForCollisions(
        nameToFieldMap,
        "no" + optionName,
        optionDefinition,
        "boolean option alias",
        allowDuplicatesParsingEquivalently);
    checkForCollisions(
        oldNameToFieldMap,
        "no" + optionName,
        optionDefinition,
        "boolean option alias",
        allowDuplicatesParsingEquivalently);

    // Record that the boolean option takes up additional namespace for its negating alias.
    booleanAliasMap.put("no" + optionName, optionName);
  }

  /**
   * Constructs an {@link IsolatedOptionsData} object for a parser that knows about the given {@link
   * OptionsBase} classes. No inter-option analysis is done. Performs basic validity checks on each
   * option in isolation.
   *
   * <p>If {@code allowDuplicatesParsingEquivalently} is true, then options that collide in name but
   * parse equivalently (e.g. both of them accept a value or both of them do not), are allowed.
   */
  static IsolatedOptionsData from(
      Collection<Class<? extends OptionsBase>> classes,
      boolean allowDuplicatesParsingEquivalently) {
    // Mind which fields have to preserve order.
    Map<Class<? extends OptionsBase>, Constructor<?>> constructorBuilder = new LinkedHashMap<>();
    ImmutableList.Builder<Class<? extends OptionsBase>> primaryOptionsClassesBuilder =
        ImmutableList.builder();
    Map<String, OptionDefinition> nameToFieldBuilder = new LinkedHashMap<>();
    Map<String, OptionDefinition> oldNameToFieldBuilder = new LinkedHashMap<>();
    Map<Character, OptionDefinition> abbrevToFieldBuilder = new HashMap<>();

    // Maps the negated boolean flag aliases to the original option name.
    Map<String, String> booleanAliasMap = new HashMap<>();

    // Combine the option definitions for these options classes, and check that they do not
    // conflict. The options are individually checked for correctness at compile time in the
    // OptionProcessor.
    for (Class<? extends OptionsBase> parsedOptionsClass : classes) {
      primaryOptionsClassesBuilder.add(parsedOptionsClass);
      Constructor<? extends OptionsBase> constructor;
      try {
        Class<? extends OptionsBase> classToInstantiate = parsedOptionsClass;
        if (parsedOptionsClass.isAnnotationPresent(OptionsClass.class)) {
          classToInstantiate = MethodOptionDefinition.getImplClass(parsedOptionsClass);
        }
        constructor = classToInstantiate.getConstructor();
        constructorBuilder.put(parsedOptionsClass, constructor);
      } catch (NoSuchMethodException e) {
        throw new IllegalArgumentException(
            parsedOptionsClass + " lacks an accessible default constructor", e);
      }

      for (Class<? extends OptionsBase> superclass : getAllSuperclasses(parsedOptionsClass)) {
        // If two options classes have the same base class or one is the base class of another,
        // it's an option conflict. Except for fallback options (when
        // allowDuplicatesParsingEquivalently is true), but then we don't instantiate any option
        // classes
        if (constructorBuilder.containsKey(superclass)) {
          constructorBuilder.put(superclass, AMBIGUOUS_MARKER_CTOR);
        } else {
          constructorBuilder.put(superclass, constructor);
        }
      }

      ImmutableList<OptionDefinition> optionDefinitions =
          getAllOptionDefinitionsForClass(parsedOptionsClass);

      for (OptionDefinition optionDefinition : optionDefinitions) {
        try {
          String optionName = optionDefinition.getOptionName();
          checkForCollisions(
              nameToFieldBuilder,
              optionName,
              optionDefinition,
              "option name collision",
              allowDuplicatesParsingEquivalently);
          checkForCollisions(
              oldNameToFieldBuilder,
              optionName,
              optionDefinition,
              "option name collision with another option's old name",
              allowDuplicatesParsingEquivalently);
          checkForBooleanAliasCollisions(booleanAliasMap, optionName, "option");
          if (optionDefinition.usesBooleanValueSyntax()) {
            checkAndUpdateBooleanAliases(
                nameToFieldBuilder,
                oldNameToFieldBuilder,
                booleanAliasMap,
                optionName,
                optionDefinition,
                allowDuplicatesParsingEquivalently);
          }
          nameToFieldBuilder.put(optionName, optionDefinition);

          if (!optionDefinition.getOldOptionName().isEmpty()) {
            String oldName = optionDefinition.getOldOptionName();
            checkForCollisions(
                nameToFieldBuilder,
                oldName,
                optionDefinition,
                "old option name collision with another option's canonical name",
                allowDuplicatesParsingEquivalently);
            checkForCollisions(
                oldNameToFieldBuilder,
                oldName,
                optionDefinition,
                "old option name collision with another old option name",
                allowDuplicatesParsingEquivalently);
            checkForBooleanAliasCollisions(booleanAliasMap, oldName, "old option name");
            // If boolean, repeat the alias dance for the old name.
            if (optionDefinition.usesBooleanValueSyntax()) {
              checkAndUpdateBooleanAliases(
                  nameToFieldBuilder,
                  oldNameToFieldBuilder,
                  booleanAliasMap,
                  oldName,
                  optionDefinition,
                  allowDuplicatesParsingEquivalently);
            }
            // Now that we've checked for conflicts, confidently store the old name.
            oldNameToFieldBuilder.put(oldName, optionDefinition);
          }
          if (optionDefinition.getAbbreviation() != '\0') {
            checkForCollisions(
                abbrevToFieldBuilder,
                optionDefinition.getAbbreviation(),
                optionDefinition,
                "option abbreviation",
                allowDuplicatesParsingEquivalently);
            abbrevToFieldBuilder.put(optionDefinition.getAbbreviation(), optionDefinition);
          }
        } catch (DuplicateOptionDeclarationException e) {
          throw new ConstructionException(e);
        }
      }
    }

    return new IsolatedOptionsData(
        constructorBuilder,
        primaryOptionsClassesBuilder.build(),
        nameToFieldBuilder,
        oldNameToFieldBuilder,
        abbrevToFieldBuilder);
  }

  private static ImmutableSet<Class<? extends OptionsBase>> getAllSuperclasses(
      Class<? extends OptionsBase> clazz) {
    ImmutableSet.Builder<Class<? extends OptionsBase>> builder = ImmutableSet.builder();
    Class<?> current = clazz.getSuperclass();
    // We don't check for nullness because every class here should be a descendant of OptionsBase
    while (current != OptionsBase.class) {
      builder.add(current.asSubclass(OptionsBase.class));
      current = current.getSuperclass();
    }
    return builder.build();
  }
}
