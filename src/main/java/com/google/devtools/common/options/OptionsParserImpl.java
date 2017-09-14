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

import static java.util.Comparator.comparing;
import static java.util.stream.Collectors.toCollection;

import com.google.common.base.Joiner;
import com.google.common.base.Preconditions;
import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.ImmutableList;
import com.google.common.collect.Iterators;
import com.google.common.collect.LinkedHashMultimap;
import com.google.common.collect.Multimap;
import com.google.devtools.common.options.OptionsParser.OptionDescription;
import java.lang.reflect.Constructor;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.function.Function;
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
   *   OptionDefinition("--host") -> "www.google.com"
   *   OptionDefinition("--port") -> 80
   * </pre>
   *
   * This map is modified by repeated calls to {@link #parse(OptionPriority,Function,List)}.
   */
  private final Map<OptionDefinition, OptionValueDescription> parsedValues = new HashMap<>();

  /**
   * We store the pre-parsed, explicit options for each priority in here.
   * We use partially preparsed options, which can be different from the original
   * representation, e.g. "--nofoo" becomes "--foo=0".
   */
  private final List<UnparsedOptionValueDescription> unparsedValues = new ArrayList<>();

  /**
   * Unparsed values for use with the canonicalize command are stored separately from unparsedValues
   * so that invocation policy can modify the values for canonicalization (e.g. override
   * user-specified values with default values) without corrupting the data used to represent the
   * user's original invocation for {@link #asListOfExplicitOptions()} and {@link
   * #asListOfUnparsedOptions()}. A LinkedHashMultimap is used so that canonicalization happens in
   * the correct order and multiple values can be stored for flags that allow multiple values.
   */
  private final Multimap<OptionDefinition, UnparsedOptionValueDescription> canonicalizeValues =
      LinkedHashMultimap.create();

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
    return unparsedValues
        .stream()
        // It is vital that this sort is stable so that options on the same priority are not
        // reordered.
        .sorted(comparing(UnparsedOptionValueDescription::getPriority))
        .collect(toCollection(ArrayList::new));
  }

  /**
   * Implements {@link OptionsParser#asListOfExplicitOptions()}.
   */
  List<UnparsedOptionValueDescription> asListOfExplicitOptions() {
    return unparsedValues
        .stream()
        .filter(UnparsedOptionValueDescription::isExplicit)
        // It is vital that this sort is stable so that options on the same priority are not
        // reordered.
        .sorted(comparing(UnparsedOptionValueDescription::getPriority))
        .collect(toCollection(ArrayList::new));
  }

  /**
   * Implements {@link OptionsParser#canonicalize}.
   */
  List<String> asCanonicalizedList() {
    return canonicalizeValues
        .values()
        .stream()
        // Sort implicit requirement options to the end, keeping their existing order, and sort
        // the other options alphabetically.
        .sorted(
            (v1, v2) -> {
              if (v1.isImplicitRequirement()) {
                return v2.isImplicitRequirement() ? 0 : 1;
              }
              if (v2.isImplicitRequirement()) {
                return -1;
              }
              return v1.getOptionDefinition()
                  .getOptionName()
                  .compareTo(v2.getOptionDefinition().getOptionName());
            })
        // Ignore expansion options.
        .filter(value -> !value.isExpansion())
        .map(
            value ->
                "--"
                    + value.getOptionDefinition().getOptionName()
                    + "="
                    + value.getUnconvertedValue())
        .collect(toCollection(ArrayList::new));
  }

  /**
   * Implements {@link OptionsParser#asListOfEffectiveOptions()}.
   */
  List<OptionValueDescription> asListOfEffectiveOptions() {
    List<OptionValueDescription> result = new ArrayList<>();
    for (Map.Entry<String, OptionDefinition> mapEntry : optionsData.getAllOptionDefinitions()) {
      OptionDefinition optionDefinition = mapEntry.getValue();
      OptionValueDescription entry = parsedValues.get(optionDefinition);
      if (entry == null) {
        result.add(OptionValueDescription.newDefaultValue(optionDefinition));
      } else {
        result.add(entry);
      }
    }
    return result;
  }

  private void maybeAddDeprecationWarning(OptionDefinition optionDefinition) {
    // Continue to support the old behavior for @Deprecated options.
    String warning = optionDefinition.getDeprecationWarning();
    if (!warning.isEmpty() || (optionDefinition.getField().isAnnotationPresent(Deprecated.class))) {
      addDeprecationWarning(optionDefinition.getOptionName(), warning);
    }
  }

  private void addDeprecationWarning(String optionName, String warning) {
    warnings.add("Option '" + optionName + "' is deprecated"
        + (warning.isEmpty() ? "" : ": " + warning));
  }

  // Warnings should not end with a '.' because the internal reporter adds one automatically.
  private void setValue(
      UnparsedOptionValueDescription optionValue,
      OptionDefinition implicitDependant,
      OptionDefinition expandedFrom)
      throws OptionsParsingException {
    OptionDefinition optionDefinition = optionValue.getOptionDefinition();
    Preconditions.checkArgument(!optionDefinition.allowsMultiple());
    Object convertedValue = optionValue.getConvertedValue();
    OptionValueDescription entry = parsedValues.get(optionValue.getOptionDefinition());
    if (entry != null) {
      // Override existing option if the new value has higher or equal priority.
      if (optionValue.getPriority().compareTo(entry.getPriority()) >= 0) {
        // Output warnings:
        if ((implicitDependant != null) && (entry.getImplicitDependant() != null)) {
          if (!implicitDependant.equals(entry.getImplicitDependant())) {
            warnings.add(
                "Option '"
                    + optionDefinition.getOptionName()
                    + "' is implicitly defined by both option '"
                    + entry.getImplicitDependant().getOptionName()
                    + "' and option '"
                    + implicitDependant.getOptionName()
                    + "'");
          }
        } else if ((implicitDependant != null)
            && optionValue.getPriority().equals(entry.getPriority())) {
          warnings.add(
              "Option '"
                  + optionDefinition.getOptionName()
                  + "' is implicitly defined by option '"
                  + implicitDependant.getOptionName()
                  + "'; the implicitly set value overrides the previous one");
        } else if (entry.getImplicitDependant() != null) {
          warnings.add(
              "A new value for option '"
                  + optionDefinition.getOptionName()
                  + "' overrides a previous implicit setting of that option by option '"
                  + entry.getImplicitDependant().getOptionName()
                  + "'");
        } else if ((optionValue.getPriority() == entry.getPriority())
            && ((entry.getExpansionParent() == null) && (expandedFrom != null))) {
          // Create a warning if an expansion option overrides an explicit option:
          warnings.add(
              "The option '"
                  + expandedFrom.getOptionName()
                  + "' was expanded and now overrides a "
                  + "previous explicitly specified option '"
                  + optionDefinition.getOptionName()
                  + "'");
        } else if ((entry.getExpansionParent() != null) && (expandedFrom != null)) {
          warnings.add(
              "The option '"
                  + optionDefinition.getOptionName()
                  + "' was expanded to from both options '"
                  + entry.getExpansionParent().getOptionName()
                  + "' and '"
                  + expandedFrom.getOptionName()
                  + "'");
        }

        // Record the new value:
        parsedValues.put(
            optionDefinition,
            OptionValueDescription.newOptionValue(
                optionDefinition,
                null,
                convertedValue,
                optionValue.getPriority(),
                optionValue.getSource(),
                implicitDependant,
                expandedFrom));
      }
    } else {
      parsedValues.put(
          optionDefinition,
          OptionValueDescription.newOptionValue(
              optionDefinition,
              null,
              convertedValue,
              optionValue.getPriority(),
              optionValue.getSource(),
              implicitDependant,
              expandedFrom));
      maybeAddDeprecationWarning(optionDefinition);
    }
  }

  private void addListValue(
      UnparsedOptionValueDescription optionValue,
      OptionDefinition implicitDependant,
      OptionDefinition expandedFrom)
      throws OptionsParsingException {
    OptionDefinition optionDefinition = optionValue.getOptionDefinition();
    Preconditions.checkArgument(optionDefinition.allowsMultiple());

    OptionValueDescription entry = parsedValues.get(optionDefinition);
    if (entry == null) {
      entry =
          OptionValueDescription.newOptionValue(
              optionDefinition,
              /* originalValueString */ null,
              ArrayListMultimap.create(),
              optionValue.getPriority(),
              optionValue.getSource(),
              implicitDependant,
              expandedFrom);
      parsedValues.put(optionDefinition, entry);
      maybeAddDeprecationWarning(optionDefinition);
    }
    Object convertedValue = optionValue.getConvertedValue();
    entry.addValue(optionValue.getPriority(), convertedValue);
  }

  OptionValueDescription clearValue(OptionDefinition optionDefinition)
      throws OptionsParsingException {
    // Actually remove the value from various lists tracking effective options.
    canonicalizeValues.removeAll(optionDefinition);
    return parsedValues.remove(optionDefinition);
  }

  OptionValueDescription getOptionValueDescription(String name) {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return parsedValues.get(optionDefinition);
  }

  OptionDescription getOptionDescription(String name, OptionPriority priority, String source)
      throws OptionsParsingException {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      return null;
    }

    return new OptionDescription(
        optionDefinition,
        optionsData.getExpansionDataForField(optionDefinition),
        getImplicitDependantDescriptions(
            ImmutableList.copyOf(optionDefinition.getImplicitRequirements()),
            optionDefinition,
            priority,
            source));
  }

  /** @return A list of the descriptions corresponding to the implicit dependant flags passed in. */
  private ImmutableList<UnparsedOptionValueDescription> getImplicitDependantDescriptions(
      ImmutableList<String> options,
      OptionDefinition implicitDependant,
      OptionPriority priority,
      String source)
      throws OptionsParsingException {
    ImmutableList.Builder<UnparsedOptionValueDescription> builder = ImmutableList.builder();
    Iterator<String> optionsIterator = options.iterator();

    Function<OptionDefinition, String> sourceFunction =
        o ->
            String.format(
                "implicitely required for option %s (source: %s)",
                implicitDependant.getOptionName(), source);
    while (optionsIterator.hasNext()) {
      String unparsedFlagExpression = optionsIterator.next();
      UnparsedOptionValueDescription unparsedOption =
          identifyOptionAndPossibleArgument(
              unparsedFlagExpression, optionsIterator, priority, sourceFunction, false);
      builder.add(unparsedOption);
    }
    return builder.build();
  }

  /**
   * @return A list of the descriptions corresponding to options expanded from the flag for the
   *     given value. These descriptions are are divorced from the command line - there is no
   *     correct priority or source for these, as they are not actually set values. The value itself
   *     is also a string, no conversion has taken place.
   */
  ImmutableList<UnparsedOptionValueDescription> getExpansionOptionValueDescriptions(
      OptionDefinition expansionFlag,
      @Nullable String flagValue,
      OptionPriority priority,
      String source)
      throws OptionsParsingException {
    ImmutableList.Builder<UnparsedOptionValueDescription> builder = ImmutableList.builder();

    ImmutableList<String> options = optionsData.getEvaluatedExpansion(expansionFlag, flagValue);
    Iterator<String> optionsIterator = options.iterator();
    Function<OptionDefinition, String> sourceFunction =
        o -> String.format("expanded from %s (source: %s)", expansionFlag.getOptionName(), source);
    while (optionsIterator.hasNext()) {
      String unparsedFlagExpression = optionsIterator.next();
      UnparsedOptionValueDescription unparsedOption =
          identifyOptionAndPossibleArgument(
              unparsedFlagExpression, optionsIterator, priority, sourceFunction, false);
      builder.add(unparsedOption);
    }
    return builder.build();
  }

  boolean containsExplicitOption(String name) {
    OptionDefinition optionDefinition = optionsData.getOptionDefinitionFromName(name);
    if (optionDefinition == null) {
      throw new IllegalArgumentException("No such option '" + name + "'");
    }
    return parsedValues.get(optionDefinition) != null;
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple times, and may be
   * called recursively. In each call, there may be no duplicates, but separate calls may contain
   * intersecting sets of options; in that case, the arg seen last takes precedence.
   */
  List<String> parse(
      OptionPriority priority, Function<OptionDefinition, String> sourceFunction, List<String> args)
      throws OptionsParsingException {
    return parse(priority, sourceFunction, null, null, args);
  }

  /**
   * Parses the args, and returns what it doesn't parse. May be called multiple times, and may be
   * called recursively. Calls may contain intersecting sets of options; in that case, the arg seen
   * last takes precedence.
   *
   * <p>The method uses the invariant that if an option has neither an implicit dependent nor an
   * expanded from value, then it must have been explicitly set.
   */
  private List<String> parse(
      OptionPriority priority,
      Function<OptionDefinition, String> sourceFunction,
      OptionDefinition implicitDependent,
      OptionDefinition expandedFrom,
      List<String> args)
      throws OptionsParsingException {
    boolean isExplicit = expandedFrom == null && implicitDependent == null;
    List<String> unparsedArgs = new ArrayList<>();
    LinkedHashMap<OptionDefinition, List<String>> implicitRequirements = new LinkedHashMap<>();

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

      UnparsedOptionValueDescription unparsedOption =
          identifyOptionAndPossibleArgument(
              arg, argsIterator, priority, sourceFunction, isExplicit);
      OptionDefinition optionDefinition = unparsedOption.getOptionDefinition();
      @Nullable String unconvertedValue = unparsedOption.getUnconvertedValue();

      if (optionDefinition.isWrapperOption()) {
        if (unconvertedValue.startsWith("-")) {
          String sourceMessage =
              "Unwrapped from wrapper option --" + optionDefinition.getOptionName();
          List<String> unparsed =
              parse(
                  priority,
                  o -> sourceMessage,
                  null, // implicitDependent
                  null, // expandedFrom
                  ImmutableList.of(unconvertedValue));

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
          throw new OptionsParsingException(
              "Invalid --"
                  + optionDefinition.getOptionName()
                  + " value format. "
                  + "You may have meant --"
                  + optionDefinition.getOptionName()
                  + "=--"
                  + unconvertedValue);
        }
      }

      if (implicitDependent == null) {
        // Log explicit options and expanded options in the order they are parsed (can be sorted
        // later). Also remember whether they were expanded or not. This information is needed to
        // correctly canonicalize flags.
        unparsedValues.add(unparsedOption);
        if (optionDefinition.allowsMultiple()) {
          canonicalizeValues.put(optionDefinition, unparsedOption);
        } else {
          canonicalizeValues.replaceValues(optionDefinition, ImmutableList.of(unparsedOption));
        }
      }

      // Handle expansion options.
      if (optionDefinition.isExpansionOption()) {
        ImmutableList<String> expansion =
            optionsData.getEvaluatedExpansion(optionDefinition, unconvertedValue);

        String sourceMessage =
            "expanded from option --"
                + optionDefinition.getOptionName()
                + " from "
                + sourceFunction.apply(optionDefinition);
        Function<OptionDefinition, String> expansionSourceFunction = o -> sourceMessage;
        maybeAddDeprecationWarning(optionDefinition);
        List<String> unparsed =
            parse(priority, expansionSourceFunction, null, optionDefinition, expansion);
        if (!unparsed.isEmpty()) {
          // Throw an assertion, because this indicates an error in the definition of this
          // option's expansion, not with the input as provided by the user.
          throw new AssertionError(
              "Unparsed options remain after parsing expansion of "
                  + arg
                  + ": "
                  + Joiner.on(' ').join(unparsed));
        }
      } else {
        // ...but allow duplicates of single-use options across separate calls to
        // parse(); latest wins:
        if (!optionDefinition.allowsMultiple()) {
          setValue(unparsedOption, implicitDependent, expandedFrom);
        } else {
          // But if it's a multiple-use option, then accumulate the values, in the order in which
          // they were seen.
          addListValue(unparsedOption, implicitDependent, expandedFrom);
        }
      }

      // Collect any implicit requirements.
      if (optionDefinition.getImplicitRequirements().length > 0) {
        implicitRequirements.put(
            optionDefinition, Arrays.asList(optionDefinition.getImplicitRequirements()));
      }
    }

    // Now parse any implicit requirements that were collected.
    // TODO(bazel-team): this should happen when the option is encountered.
    if (!implicitRequirements.isEmpty()) {
      for (Map.Entry<OptionDefinition, List<String>> entry : implicitRequirements.entrySet()) {
        String sourceMessage =
            "implicit requirement of option --"
                + entry.getKey()
                + " from "
                + sourceFunction.apply(entry.getKey());
        Function<OptionDefinition, String> requirementSourceFunction = o -> sourceMessage;

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

  private UnparsedOptionValueDescription identifyOptionAndPossibleArgument(
      String arg,
      Iterator<String> nextArgs,
      OptionPriority priority,
      Function<OptionDefinition, String> sourceFunction,
      boolean explicit)
      throws OptionsParsingException {

    // Store the way this option was parsed on the command line.
    StringBuilder commandLineForm = new StringBuilder();
    commandLineForm.append(arg);
    String unparsedValue = null;
    OptionDefinition optionDefinition;
    boolean booleanValue = true;

    if (arg.length() == 2) { // -l  (may be nullary or unary)
      optionDefinition = optionsData.getFieldForAbbrev(arg.charAt(1));
      booleanValue = true;

    } else if (arg.length() == 3 && arg.charAt(2) == '-') { // -l-  (boolean)
      optionDefinition = optionsData.getFieldForAbbrev(arg.charAt(1));
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
      unparsedValue = equalsAt == -1 ? null : arg.substring(equalsAt + 1);
      optionDefinition = optionsData.getOptionDefinitionFromName(name);

      // Look for a "no"-prefixed option name: "no<optionName>".
      if (optionDefinition == null && name.startsWith("no")) {
        name = name.substring(2);
        optionDefinition = optionsData.getOptionDefinitionFromName(name);
        booleanValue = false;
        if (optionDefinition != null) {
          // TODO(bazel-team): Add tests for these cases.
          if (!optionDefinition.usesBooleanValueSyntax()) {
            throw new OptionsParsingException(
                "Illegal use of 'no' prefix on non-boolean option: " + arg, arg);
          }
          if (unparsedValue != null) {
            throw new OptionsParsingException(
                "Unexpected value after boolean option: " + arg, arg);
          }
          // "no<optionname>" signifies a boolean option w/ false value
          unparsedValue = "0";
        }
      }
    } else {
      throw new OptionsParsingException("Invalid options syntax: " + arg, arg);
    }

    if (optionDefinition == null
        || ImmutableList.copyOf(optionDefinition.getOptionMetadataTags())
            .contains(OptionMetadataTag.INTERNAL)) {
      // Do not recognize internal options, which are treated as if they did not exist.
      throw new OptionsParsingException("Unrecognized option: " + arg, arg);
    }

    if (unparsedValue == null) {
      // Special-case boolean to supply value based on presence of "no" prefix.
      if (optionDefinition.usesBooleanValueSyntax()) {
        unparsedValue = booleanValue ? "1" : "0";
      } else if (optionDefinition.getType().equals(Void.class)
          && !optionDefinition.isWrapperOption()) {
        // This is expected, Void type options have no args (unless they're wrapper options).
      } else if (nextArgs.hasNext()) {
        // "--flag value" form
        unparsedValue = nextArgs.next();
        commandLineForm.append(" ").append(unparsedValue);
      } else {
        throw new OptionsParsingException("Expected value after " + arg);
      }
    }

    return new UnparsedOptionValueDescription(
        optionDefinition,
        commandLineForm.toString(),
        unparsedValue,
        priority,
        sourceFunction.apply(optionDefinition),
        explicit);
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
    for (OptionDefinition optionDefinition :
        OptionsData.getAllOptionDefinitionsForClass(optionsClass)) {
      Object value;
      OptionValueDescription entry = parsedValues.get(optionDefinition);
      if (entry == null) {
        value = optionDefinition.getDefaultValue();
      } else {
        value = entry.getValue();
      }
      try {
        optionDefinition.getField().set(optionsInstance, value);
      } catch (IllegalAccessException e) {
        throw new IllegalStateException(e);
      }
    }
    return optionsInstance;
  }

  List<String> getWarnings() {
    return ImmutableList.copyOf(warnings);
  }
}
