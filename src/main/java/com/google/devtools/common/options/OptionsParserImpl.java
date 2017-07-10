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

import com.google.common.base.Function;
import com.google.common.base.Functions;
import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.base.Predicate;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterables;
import com.google.common.collect.Iterators;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import com.google.devtools.common.options.OptionsParser.OptionDescription;
import com.google.devtools.common.options.OptionsParser.OptionUsageRestrictions;
import com.google.devtools.common.options.OptionsParser.OptionValueDescription;
import com.google.devtools.common.options.OptionsParser.UnparsedOptionValueDescription;
import java.lang.reflect.Constructor;
import java.lang.reflect.Field;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import javax.annotation.Nullable;

/**
 * The implementation of the options parser. This is intentionally package
 * private for full flexibility. Use {@link OptionsParser} or {@link Options}
 * if you're a consumer.
 */
class OptionsParserImpl {

  private final OptionsData optionsData;

  /**
   * We store the results of parsing the arguments in here. It'll look like
   *
   * <pre>
   *   Field("--host") -> "www.google.com"
   *   Field("--port") -> 80
   * </pre>
   *
   * This map is modified by repeated calls to {@link #parse(OptionPriority,Function,List)}.
   */
  private final Map<Field, OptionValueDescription> parsedValues = new HashMap<>();

  /**
   * We store the pre-parsed, explicit options for each priority in here.
   * We use partially preparsed options, which can be different from the original
   * representation, e.g. "--nofoo" becomes "--foo=0".
   */
  private final List<UnparsedOptionValueDescription> unparsedValues = new ArrayList<>();

  /**
   * Unparsed values for use with the canonicalize command are stored separately from
   * unparsedValues so that invocation policy can modify the values for canonicalization (e.g.
   * override user-specified values with default values) without corrupting the data used to
   * represent the user's original invocation for {@link #asListOfExplicitOptions()} and
   * {@link #asListOfUnparsedOptions()}. A LinkedHashMultimap is used so that canonicalization
   * happens in the correct order and multiple values can be stored for flags that allow multiple
   * values.
   */
  private final Multimap<Field, UnparsedOptionValueDescription> canonicalizeValues
      = LinkedHashMultimap.create();

  private final List<String> warnings = new ArrayList<>();

  private boolean allowSingleDashLongOptions = false;

  private ArgsPreProcessor argsPreProcessor =
      new ArgsPreProcessor() {
        @Override
        public List<String> preProcess(List<String> args) throws OptionsParsingException {
          return args;
        }
      };

  /**
   * Create a new parser object
   */
  OptionsParserImpl(OptionsData optionsData) {
    this.optionsData = optionsData;
  }

  OptionsData getOptionsData() {
    return optionsData;
  }

  /**
   * Indicates whether or not the parser will allow long options with a
   * single-dash, instead of the usual double-dash, too, eg. -example instead of just --example.
   */
  void setAllowSingleDashLongOptions(boolean allowSingleDashLongOptions) {
    this.allowSingleDashLongOptions = allowSingleDashLongOptions;
  }

  /** Sets the ArgsPreProcessor for manipulations of the options before parsing. */
  void setArgsPreProcessor(ArgsPreProcessor preProcessor) {
    this.argsPreProcessor = Preconditions.checkNotNull(preProcessor);
  }

  /**
   * Implements {@link OptionsParser#asListOfUnparsedOptions()}.
   */
  List<UnparsedOptionValueDescription> asListOfUnparsedOptions() {
    List<UnparsedOptionValueDescription> result = Lists.newArrayList(unparsedValues);
    // It is vital that this sort is stable so that options on the same priority are not reordered.
    Collections.sort(result, new Comparator<UnparsedOptionValueDescription>() {
      @Override
      public int compare(UnparsedOptionValueDescription o1,
          UnparsedOptionValueDescription o2) {
        return o1.getPriority().compareTo(o2.getPriority());
      }
    });
    return result;
  }

  /**
   * Implements {@link OptionsParser#asListOfExplicitOptions()}.
   */
  List<UnparsedOptionValueDescription> asListOfExplicitOptions() {
    List<UnparsedOptionValueDescription> result = Lists.newArrayList(Iterables.filter(
      unparsedValues,
      new Predicate<UnparsedOptionValueDescription>() {
        @Override
        public boolean apply(UnparsedOptionValueDescription input) {
          return input.isExplicit();
        }
    }));
    // It is vital that this sort is stable so that options on the same priority are not reordered.
    Collections.sort(result, new Comparator<UnparsedOptionValueDescription>() {
      @Override
      public int compare(UnparsedOptionValueDescription o1,
          UnparsedOptionValueDescription o2) {
        return o1.getPriority().compareTo(o2.getPriority());
      }
    });
    return result;
  }

  /**
   * Implements {@link OptionsParser#canonicalize}.
   */
  List<String> asCanonicalizedList() {

    List<UnparsedOptionValueDescription> processed = Lists.newArrayList(
        canonicalizeValues.values());
    // Sort implicit requirement options to the end, keeping their existing order, and sort the
    // other options alphabetically.
    Collections.sort(processed, new Comparator<UnparsedOptionValueDescription>() {
      @Override
      public int compare(UnparsedOptionValueDescription o1, UnparsedOptionValueDescription o2) {
        if (o1.isImplicitRequirement()) {
          return o2.isImplicitRequirement() ? 0 : 1;
        }
        if (o2.isImplicitRequirement()) {
          return -1;
        }
        return o1.getName().compareTo(o2.getName());
      }
    });

    List<String> result = new ArrayList<>();
    for (UnparsedOptionValueDescription value : processed) {

      // Ignore expansion options.
      if (value.isExpansion()) {
        continue;
      }

      result.add("--" + value.getName() + "=" + value.getUnparsedValue());
    }
    return result;
  }

  /**
   * Implements {@link OptionsParser#asListOfEffectiveOptions()}.
   */
  List<OptionValueDescription> asListOfEffectiveOptions() {
    List<OptionValueDescription> result = new ArrayList<>();
    for (Map.Entry<String, Field> mapEntry : optionsData.getAllNamedFields()) {
      String fieldName = mapEntry.getKey();
      Field field = mapEntry.getValue();
      OptionValueDescription entry = parsedValues.get(field);
      if (entry == null) {
        Object value = optionsData.getDefaultValue(field);
        result.add(
            new OptionValueDescription(
                fieldName,
                /*originalValueString=*/ null,
                value,
                OptionPriority.DEFAULT,
                /*source=*/ null,
                /*implicitDependant=*/ null,
                /*expandedFrom=*/ null,
                false));
      } else {
        result.add(entry);
      }
    }
    return result;
  }

  private void maybeAddDeprecationWarning(Field field) {
    Option option = field.getAnnotation(Option.class);
    // Continue to support the old behavior for @Deprecated options.
    String warning = option.deprecationWarning();
    if (!warning.isEmpty() || (field.getAnnotation(Deprecated.class) != null)) {
      addDeprecationWarning(option.name(), warning);
    }
  }

  private void addDeprecationWarning(String optionName, String warning) {
    warnings.add("Option '" + optionName + "' is deprecated"
        + (warning.isEmpty() ? "" : ": " + warning));
  }

  // Warnings should not end with a '.' because the internal reporter adds one automatically.
  private void setValue(Field field, String name, Object value,
      OptionPriority priority, String source, String implicitDependant, String expandedFrom) {
    OptionValueDescription entry = parsedValues.get(field);
    if (entry != null) {
      // Override existing option if the new value has higher or equal priority.
      if (priority.compareTo(entry.getPriority()) >= 0) {
        // Output warnings:
        if ((implicitDependant != null) && (entry.getImplicitDependant() != null)) {
          if (!implicitDependant.equals(entry.getImplicitDependant())) {
            warnings.add(
                "Option '"
                    + name
                    + "' is implicitly defined by both option '"
                    + entry.getImplicitDependant()
                    + "' and option '"
                    + implicitDependant
                    + "'");
          }
        } else if ((implicitDependant != null) && priority.equals(entry.getPriority())) {
          warnings.add(
              "Option '"
                  + name
                  + "' is implicitly defined by option '"
                  + implicitDependant
                  + "'; the implicitly set value overrides the previous one");
        } else if (entry.getImplicitDependant() != null) {
          warnings.add(
              "A new value for option '"
                  + name
                  + "' overrides a previous implicit setting of that option by option '"
                  + entry.getImplicitDependant()
                  + "'");
        } else if ((priority == entry.getPriority())
            && ((entry.getExpansionParent() == null) && (expandedFrom != null))) {
          // Create a warning if an expansion option overrides an explicit option:
          warnings.add("The option '" + expandedFrom + "' was expanded and now overrides a "
              + "previous explicitly specified option '" + name + "'");
        } else if ((entry.getExpansionParent() != null) && (expandedFrom != null)) {
          warnings.add(
              "The option '"
                  + name
                  + "' was expanded to from both options '"
                  + entry.getExpansionParent()
                  + "' and '"
                  + expandedFrom
                  + "'");
        }

        // Record the new value:
        parsedValues.put(
            field,
            new OptionValueDescription(
                name, null, value, priority, source, implicitDependant, expandedFrom, false));
      }
    } else {
      parsedValues.put(
          field,
          new OptionValueDescription(
              name, null, value, priority, source, implicitDependant, expandedFrom, false));
      maybeAddDeprecationWarning(field);
    }
  }

  private void addListValue(Field field, String originalName, Object value, OptionPriority priority,
      String source, String implicitDependant, String expandedFrom) {
    OptionValueDescription entry = parsedValues.get(field);
    if (entry == null) {
      entry =
          new OptionValueDescription(
              originalName,
              /* originalValueString */ null,
              ArrayListMultimap.create(),
              priority,
              source,
              implicitDependant,
              expandedFrom,
              true);
      parsedValues.put(field, entry);
      maybeAddDeprecationWarning(field);
    }
    entry.addValue(priority, value);
  }

  OptionValueDescription clearValue(String optionName)
      throws OptionsParsingException {
    Field field = optionsData.getFieldFromName(optionName);
    if (field == null) {
      throw new IllegalArgumentException("No such option '" + optionName + "'");
    }

    // Actually remove the value from various lists tracking effective options.
    canonicalizeValues.removeAll(field);
    return parsedValues.remove(field);
  }

  OptionValueDescription getOptionValueDescription(String name) {
    Field field = optionsData.getFieldFromName(name);
    if (field == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return parsedValues.get(field);
  }

  OptionDescription getOptionDescription(String name) throws OptionsParsingException {
    Field field = optionsData.getFieldFromName(name);
    if (field == null) {
      return null;
    }

    Option optionAnnotation = field.getAnnotation(Option.class);
    return new OptionDescription(
        name,
        optionsData.getDefaultValue(field),
        optionsData.getConverter(field),
        optionsData.getAllowMultiple(field),
        optionsData.getExpansionDataForField(field),
        getImplicitDependantDescriptions(
            ImmutableList.copyOf(optionAnnotation.implicitRequirements()), name));
  }

  /**
   * @return A list of the descriptions corresponding to the implicit dependant flags passed in.
   *     These descriptions are are divorced from the command line - there is no correct priority or
   *     source for these, as they are not actually set values. The value itself is also a string,
   *     no conversion has taken place.
   */
  private ImmutableList<OptionValueDescription> getImplicitDependantDescriptions(
      ImmutableList<String> options, String implicitDependant) throws OptionsParsingException {
    ImmutableList.Builder<OptionValueDescription> builder = ImmutableList.builder();
    Iterator<String> optionsIterator = options.iterator();

    while (optionsIterator.hasNext()) {
      String unparsedFlagExpression = optionsIterator.next();
      ParseOptionResult parseResult = parseOption(unparsedFlagExpression, optionsIterator);
      builder.add(
          new OptionValueDescription(
              parseResult.option.name(),
              parseResult.value,
              /* value */ null,
              /* priority */ null,
              /* source */ null,
              implicitDependant,
              /* expendedFrom */ null,
              optionsData.getAllowMultiple(parseResult.field)));
    }
    return builder.build();
  }

  /**
   * @return A list of the descriptions corresponding to options expanded from the flag for the
   *     given value. These descriptions are are divorced from the command line - there is no
   *     correct priority or source for these, as they are not actually set values. The value itself
   *     is also a string, no conversion has taken place.
   */
  ImmutableList<OptionValueDescription> getExpansionOptionValueDescriptions(
      String flagName, @Nullable String flagValue) throws OptionsParsingException {
    ImmutableList.Builder<OptionValueDescription> builder = ImmutableList.builder();
    Field field = optionsData.getFieldFromName(flagName);

    ImmutableList<String> options = optionsData.getEvaluatedExpansion(field, flagValue);
    Iterator<String> optionsIterator = options.iterator();

    while (optionsIterator.hasNext()) {
      String unparsedFlagExpression = optionsIterator.next();
      ParseOptionResult parseResult = parseOption(unparsedFlagExpression, optionsIterator);
      builder.add(
          new OptionValueDescription(
              parseResult.option.name(),
              parseResult.value,
              /* value */ null,
              /* priority */ null,
              /* source */ null,
              /* implicitDependant */ null,
              flagName,
              optionsData.getAllowMultiple(parseResult.field)));
    }
    return builder.build();
  }

  boolean containsExplicitOption(String name) {
    Field field = optionsData.getFieldFromName(name);
    if (field == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return parsedValues.get(field) != null;
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple
   * times, and may be called recursively. In each call, there may be no
   * duplicates, but separate calls may contain intersecting sets of options; in
   * that case, the arg seen last takes precedence.
   */
  List<String> parse(OptionPriority priority, Function<? super String, String> sourceFunction,
      List<String> args) throws OptionsParsingException {
    return parse(priority, sourceFunction, null, null, args);
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple
   * times, and may be called recursively. Calls may contain intersecting sets
   * of options; in that case, the arg seen last takes precedence.
   *
   * <p>The method uses the invariant that if an option has neither an implicit
   * dependent nor an expanded from value, then it must have been explicitly
   * set.
   */
  private List<String> parse(
      OptionPriority priority,
      Function<? super String, String> sourceFunction,
      String implicitDependent,
      String expandedFrom,
      List<String> args) throws OptionsParsingException {

    List<String> unparsedArgs = new ArrayList<>();
    LinkedHashMap<String, List<String>> implicitRequirements = new LinkedHashMap<>();

    Iterator<String> argsIterator = argsPreProcessor.preProcess(args).iterator();
    while (argsIterator.hasNext()) {
      String arg = argsIterator.next();

      if (!arg.startsWith("-")) {
        unparsedArgs.add(arg);
        continue;  // not an option arg
      }

      if (arg.equals("--")) {  // "--" means all remaining args aren't options
        Iterators.addAll(unparsedArgs, argsIterator);
        break;
      }

      ParseOptionResult parseOptionResult = parseOption(arg, argsIterator);
      Field field = parseOptionResult.field;
      Option option = parseOptionResult.option;
      @Nullable String value = parseOptionResult.value;

      final String originalName = option.name();

      if (option.wrapperOption()) {
        if (value.startsWith("-")) {

          List<String> unparsed = parse(
              priority,
              Functions.constant("Unwrapped from wrapper option --" + originalName),
              null, // implicitDependent
              null, // expandedFrom
              ImmutableList.of(value));

          if (!unparsed.isEmpty()) {
            throw new OptionsParsingException(
                "Unparsed options remain after unwrapping "
                    + arg
                    + ": "
                    + Joiner.on(' ').join(unparsed));
          }

          // Don't process implicitRequirements or expansions for wrapper options. In particular,
          // don't record this option in unparsedValues, so that only the wrapped option shows
          // up in canonicalized options.
          continue;

        } else {
          throw new OptionsParsingException("Invalid --" + originalName + " value format. "
              + "You may have meant --" + originalName + "=--" + value);
        }
      }

      if (implicitDependent == null) {
        // Log explicit options and expanded options in the order they are parsed (can be sorted
        // later). Also remember whether they were expanded or not. This information is needed to
        // correctly canonicalize flags.
        UnparsedOptionValueDescription unparsedOptionValueDescription =
            new UnparsedOptionValueDescription(
                originalName,
                field,
                value,
                priority,
                sourceFunction.apply(originalName),
                expandedFrom == null);
        unparsedValues.add(unparsedOptionValueDescription);
        if (option.allowMultiple()) {
          canonicalizeValues.put(field, unparsedOptionValueDescription);
        } else {
          canonicalizeValues.replaceValues(field, ImmutableList.of(unparsedOptionValueDescription));
        }
      }

      // Handle expansion options.
      if (OptionsData.isExpansionOption(field.getAnnotation(Option.class))) {
        ImmutableList<String> expansion = optionsData.getEvaluatedExpansion(field, value);

        Function<Object, String> expansionSourceFunction =
            Functions.constant(
                "expanded from option --"
                    + originalName
                    + " from "
                    + sourceFunction.apply(originalName));
        maybeAddDeprecationWarning(field);
        List<String> unparsed =
            parse(priority, expansionSourceFunction, null, originalName, expansion);
        if (!unparsed.isEmpty()) {
          // Throw an assertion, because this indicates an error in the code that specified the
          // expansion for the current option.
          throw new AssertionError(
              "Unparsed options remain after parsing expansion of "
                  + arg
                  + ": "
                  + Joiner.on(' ').join(unparsed));
        }
      } else {
        Converter<?> converter = optionsData.getConverter(field);
        Object convertedValue;
        try {
          convertedValue = converter.convert(value);
        } catch (OptionsParsingException e) {
          // The converter doesn't know the option name, so we supply it here by
          // re-throwing:
          throw new OptionsParsingException("While parsing option " + arg
                                            + ": " + e.getMessage(), e);
        }

        // ...but allow duplicates of single-use options across separate calls to
        // parse(); latest wins:
        if (!option.allowMultiple()) {
          setValue(field, originalName, convertedValue,
              priority, sourceFunction.apply(originalName), implicitDependent, expandedFrom);
        } else {
          // But if it's a multiple-use option, then just accumulate the
          // values, in the order in which they were seen.
          // Note: The type of the list member is not known; Java introspection
          // only makes it available in String form via the signature string
          // for the field declaration.
          addListValue(field, originalName, convertedValue, priority,
              sourceFunction.apply(originalName), implicitDependent, expandedFrom);
        }
      }

      // Collect any implicit requirements.
      if (option.implicitRequirements().length > 0) {
        implicitRequirements.put(option.name(), Arrays.asList(option.implicitRequirements()));
      }
    }

    // Now parse any implicit requirements that were collected.
    // TODO(bazel-team): this should happen when the option is encountered.
    if (!implicitRequirements.isEmpty()) {
      for (Map.Entry<String, List<String>> entry : implicitRequirements.entrySet()) {
        Function<Object, String> requirementSourceFunction =
            Functions.constant(
                "implicit requirement of option --"
                    + entry.getKey()
                    + " from "
                    + sourceFunction.apply(entry.getKey()));

        List<String> unparsed = parse(priority, requirementSourceFunction, entry.getKey(), null,
            entry.getValue());
        if (!unparsed.isEmpty()) {
          // Throw an assertion, because this indicates an error in the code that specified in the
          // implicit requirements for the option(s).
          throw new AssertionError("Unparsed options remain after parsing implicit options: "
              + Joiner.on(' ').join(unparsed));
        }
      }
    }

    return unparsedArgs;
  }

  private static final class ParseOptionResult {
    final Field field;
    final Option option;
    @Nullable final String value;

    ParseOptionResult(Field field, Option option, @Nullable String value) {
      this.field = field;
      this.option = option;
      this.value = value;
    }
  }

  private ParseOptionResult parseOption(String arg, Iterator<String> nextArgs)
      throws OptionsParsingException {

    String value = null;
    Field field;
    boolean booleanValue = true;

    if (arg.length() == 2) { // -l  (may be nullary or unary)
      field = optionsData.getFieldForAbbrev(arg.charAt(1));
      booleanValue = true;

    } else if (arg.length() == 3 && arg.charAt(2) == '-') { // -l-  (boolean)
      field = optionsData.getFieldForAbbrev(arg.charAt(1));
      booleanValue = false;

    } else if (allowSingleDashLongOptions // -long_option
        || arg.startsWith("--")) { // or --long_option

      int equalsAt = arg.indexOf('=');
      int nameStartsAt = arg.startsWith("--") ? 2 : 1;
      String name =
          equalsAt == -1 ? arg.substring(nameStartsAt) : arg.substring(nameStartsAt, equalsAt);
      if (name.trim().isEmpty()) {
        throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
      }
      value = equalsAt == -1 ? null : arg.substring(equalsAt + 1);
      field = optionsData.getFieldFromName(name);

      // Look for a "no"-prefixed option name: "no<optionName>".
      if (field == null && name.startsWith("no")) {
        name = name.substring(2);
        field = optionsData.getFieldFromName(name);
        booleanValue = false;
        if (field != null) {
          // TODO(bazel-team): Add tests for these cases.
          if (!OptionsData.isBooleanField(field)) {
            throw new OptionsParsingException(
                "Illegal use of 'no' prefix on non-boolean option: " + arg, arg);
          }
          if (value != null) {
            throw new OptionsParsingException(
                "Unexpected value after boolean option: " + arg, arg);
          }
          // "no<optionname>" signifies a boolean option w/ false value
          value = "0";
        }
      }
    } else {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }

    Option option = field == null ? null : field.getAnnotation(Option.class);

    if (option == null
        || option.optionUsageRestrictions() == OptionUsageRestrictions.INTERNAL) {
      // This also covers internal options, which are treated as if they did not exist.
      throw new OptionsParsingException("Unrecognized option: " + arg, arg);
    }

    if (value == null) {
      // Special-case boolean to supply value based on presence of "no" prefix.
      if (OptionsData.isBooleanField(field)) {
        value = booleanValue ? "1" : "0";
      } else if (field.getType().equals(Void.class) && !option.wrapperOption()) {
        // This is expected, Void type options have no args (unless they're wrapper options).
      } else if (nextArgs.hasNext()) {
        value = nextArgs.next();  // "--flag value" form
      } else {
        throw new OptionsParsingException("Expected value after " + arg);
      }
    }

    return new ParseOptionResult(field, option, value);
  }

  /**
   * Gets the result of parsing the options.
   */
  <O extends OptionsBase> O getParsedOptions(Class<O> optionsClass) {
    // Create the instance:
    O optionsInstance;
    try {
      Constructor<O> constructor = optionsData.getConstructor(optionsClass);
      if (constructor == null) {
        return null;
      }
      optionsInstance = constructor.newInstance();
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException("Error while instantiating options class", e);
    }

    // Set the fields
    for (Field field : optionsData.getFieldsForClass(optionsClass)) {
      Object value;
      OptionValueDescription entry = parsedValues.get(field);
      if (entry == null) {
        value = optionsData.getDefaultValue(field);
      } else {
        value = entry.getValue();
      }
      try {
        field.set(optionsInstance, value);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }
    return optionsInstance;
  }

  List<String> getWarnings() {
    return ImmutableList.copyOf(warnings);
  }

  static String getDefaultOptionString(Field optionField) {
    Option annotation = optionField.getAnnotation(Option.class);
    return annotation.defaultValue();
  }

  static boolean isSpecialNullDefault(String defaultValueString, Field optionField) {
    return defaultValueString.equals("null") && !optionField.getType().isPrimitive();
  }
}
